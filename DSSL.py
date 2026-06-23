import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import random
# ==================== 配置 ====================
DATA_ROOT = "./Daphnet"
WINDOW = 128
STRIDE = 32
PATCH_SIZE = 16               # 必须整除 WINDOW
MASK_RATIO = 0.3
BATCH_SIZE = 64
LR_PRETRAIN = 1e-4
LR_FINETUNE = 1e-5
SSL_EPOCHS = 30
WARMUP_EPOCHS = 5
FT_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEW_SHOT_RATIOS = [0.1, 0.2, 0.5, 1.0]
NUM_REPEATS = 3
NUM_CLASSES = 2

# 物理损失超参数
PHYSICS_WEIGHT = 0.01
SSL_WEIGHT_IN_FT = 0.3
PHYSICS_SCALE = 1000.0
VISIBLE_WEIGHT = 0.15          # 可见区域重建损失的权重

# 人体参数（两连杆模型）
SEGMENT_PARAMS = {
    'thigh': {'mass': 6.0, 'length': 0.4, 'com': 0.2, 'inertia': 0.1},
    'shank': {'mass': 3.5, 'length': 0.4, 'com': 0.2, 'inertia': 0.05},
}
GRAVITY = 9.81

# ==================== 数据集 ====================
class DaphnetDataset(Dataset):
    def __init__(self, files, window=128, stride=32, norm=True):
        self.samples = []
        self.labels = []
        for file in files:
            data = np.loadtxt(file)
            x = data[:, 1:10].astype(np.float32)
            y = data[:, -1].astype(np.int64)
            y[y > 0] = 1
            if norm:
                mean = x.mean(0)
                std = x.std(0) + 1e-6
                x = (x - mean) / std
            for i in range(0, len(x) - window, stride):
                seg = x[i:i+window]
                lab = y[i:i+window]
                label = int(np.mean(lab) > 0.2)
                self.samples.append(seg)
                self.labels.append(label)
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

# ==================== Patch 工具 ====================
def patchify(x, patch_size):
    B, T, C = x.shape
    N = T // patch_size
    x_patched = x.reshape(B, N, patch_size, C).permute(0, 1, 3, 2).reshape(B, N, -1)
    return x_patched

def unpatchify(x_patched, patch_size, original_T, C):
    B, N, _ = x_patched.shape
    x = x_patched.reshape(B, N, C, patch_size).permute(0, 1, 3, 2).reshape(B, N*patch_size, C)
    return x[:, :original_T, :]

def random_masking(x_patched, mask_ratio):
    B, N, D = x_patched.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x_patched.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([B, N], device=x_patched.device, dtype=torch.bool)
    mask[:, :len_keep] = False
    mask = torch.gather(mask, dim=1, index=ids_shuffle)
    ids_keep_expand = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_masked = torch.gather(x_patched, dim=1, index=ids_keep_expand)
    return x_masked, mask, ids_shuffle, ids_keep

# ==================== 模型 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PhysGaitSSL(nn.Module):
    def __init__(self, in_channels=9, seq_len=128, patch_size=16, d_model=128, num_joints=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.d_model = d_model
        self.num_joints = num_joints

        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.ssl_head = nn.Linear(d_model, in_channels * patch_size)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )

        self.physics_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_joints),
            nn.Tanh()
        )
        self.joint_scale = 1.2

    def forward_encoder(self, x, mask_ratio=0.0):
        B, T, C = x.shape
        x_conv = x.transpose(1, 2)
        patches = self.patch_embed(x_conv).transpose(1, 2)
        patches = self.pos_enc(patches)
        mask = None
        if mask_ratio > 0.0 and self.training:
            rand = torch.rand(B, self.num_patches, device=x.device)
            mask = rand < mask_ratio
            patches[mask] = 0.0
        z = self.encoder(patches)
        return z, mask

    def forward(self, x, stage='pretrain', mask_ratio=0.3):
        z, mask = self.forward_encoder(x, mask_ratio if stage == 'pretrain' else 0.0)
        if stage == 'pretrain':
            recon_patched = self.ssl_head(z)
            recon = unpatchify(recon_patched, self.patch_size, x.shape[1], x.shape[2])
            return recon, mask
        else:
            cls_feat = z.mean(dim=1)
            logits = self.classifier(cls_feat)
            q_patched = self.physics_decoder(z)
            q = q_patched.repeat_interleave(self.patch_size, dim=1)[:, :x.shape[1], :] * self.joint_scale
            return logits, q

# ==================== 物理损失 ====================
def compute_kinematics(joint_angles, dt=0.01):
    vel = torch.zeros_like(joint_angles)
    acc = torch.zeros_like(joint_angles)
    vel[:, 1:-1, :] = (joint_angles[:, 2:, :] - joint_angles[:, :-2, :]) / (2 * dt)
    vel[:, 0, :] = (joint_angles[:, 1, :] - joint_angles[:, 0, :]) / dt
    vel[:, -1, :] = (joint_angles[:, -1, :] - joint_angles[:, -2, :]) / dt
    acc[:, 1:-1, :] = (joint_angles[:, 2:, :] - 2*joint_angles[:, 1:-1, :] + joint_angles[:, :-2, :]) / (dt**2)
    acc[:, 0, :] = (joint_angles[:, 1, :] - 2*joint_angles[:, 0, :] + joint_angles[:, 0, :]) / (dt**2)
    acc[:, -1, :] = (joint_angles[:, -1, :] - 2*joint_angles[:, -1, :] + joint_angles[:, -2, :]) / (dt**2)
    return vel, acc

def physics_loss_fn(joint_angles, dt=0.01, segment_params=SEGMENT_PARAMS, gravity=GRAVITY, scale=PHYSICS_SCALE):
    q = joint_angles[..., :2]
    vel, acc = compute_kinematics(q, dt)
    q_center = q[:, 1:-1, :]
    vel_center = vel[:, 1:-1, :]
    acc_center = acc[:, 1:-1, :]

    m1 = segment_params['thigh']['mass']; m2 = segment_params['shank']['mass']
    l1 = segment_params['thigh']['length']; l2 = segment_params['shank']['length']
    lc1 = segment_params['thigh']['com']; lc2 = segment_params['shank']['com']
    I1 = segment_params['thigh']['inertia']; I2 = segment_params['shank']['inertia']; g = gravity

    q1, q2 = q_center[..., 0], q_center[..., 1]
    dq1, dq2 = vel_center[..., 0], vel_center[..., 1]
    ddq1, ddq2 = acc_center[..., 0], acc_center[..., 1]

    M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*torch.cos(q2))
    M12 = I2 + m2*(lc2**2 + l1*lc2*torch.cos(q2))
    M22 = I2 + m2*lc2**2

    h = -m2 * l1 * lc2 * torch.sin(q2)
    C11 = 2 * h * dq2; C12 = h * dq2; C21 = -h * dq1; C22 = 0.0

    G1 = (m1*lc1 + m2*l1) * g * torch.cos(q1) + m2*lc2 * g * torch.cos(q1+q2)
    G2 = m2*lc2 * g * torch.cos(q1+q2)

    tau1 = M11*ddq1 + M12*ddq2 + C11*dq1 + C12*dq2 + G1
    tau2 = M12*ddq1 + M22*ddq2 + C21*dq1 + C22*dq2 + G2
    residual = torch.stack([tau1, tau2], dim=-1)
    loss_raw = torch.mean(residual ** 2)
    return loss_raw / scale

# ==================== 训练函数 ====================
def pretrain_stage_I(model, dataloader, optimizer, scheduler, epochs, mask_ratio, vis_weight=0.15, device='cuda'):
    model.train()
    print("=== Stage I: Self-Supervised Pretraining (Masked + Visible Loss, Cosine Annealing) ===")
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mask = model(x, stage='pretrain', mask_ratio=mask_ratio)
            mask_expanded = mask.repeat_interleave(model.patch_size, dim=1)
            # 主要损失：被 mask 区域
            loss_mask = F.mse_loss(recon[mask_expanded], x[mask_expanded])
            # 辅助损失：可见区域（小权重）
            loss_vis = F.mse_loss(recon[~mask_expanded], x[~mask_expanded])
            loss = loss_mask + vis_weight * loss_vis
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | SSL Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    print("Stage I finished.\n")

def finetune_stage_II(model, train_loader, val_loader, optimizer, epochs, warmup_epochs,
                      use_ssl=True, use_physics=True, mask_ratio=0.3, device='cuda'):
    model.train()
    print("=== Stage II: Physics-Informed Fine-tuning ===")
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for param in model.encoder.parameters(): param.requires_grad = False
            for param in model.patch_embed.parameters(): param.requires_grad = False
            phase = "Warm-up (frozen)"
        else:
            for param in model.encoder.parameters(): param.requires_grad = True
            for param in model.patch_embed.parameters(): param.requires_grad = True
            phase = "Fine-tune (unfrozen)"

        total_cls = total_phys = total_ssl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, joint_angles = model(x, stage='finetune', mask_ratio=0.0)
            loss_cls = F.cross_entropy(logits, y)
            loss_phys = physics_loss_fn(joint_angles) if use_physics else 0.0
            loss_ssl = 0.0
            if use_ssl and mask_ratio > 0:
                recon, mask = model(x, stage='pretrain', mask_ratio=mask_ratio)
                mask_expanded = mask.repeat_interleave(model.patch_size, dim=1)
                if mask.sum() > 0:
                    loss_ssl = F.mse_loss(recon[mask_expanded], x[mask_expanded])
            loss_total = loss_cls + PHYSICS_WEIGHT * loss_phys + SSL_WEIGHT_IN_FT * loss_ssl
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_cls += loss_cls.item()
            total_phys += loss_phys.item() if use_physics else 0
            total_ssl += loss_ssl.item() if use_ssl else 0

        n = len(train_loader)
        log_msg = f"Epoch {epoch+1}/{epochs} [{phase}] | Cls: {total_cls/n:.4f}"
        if use_physics: log_msg += f" | Phys: {total_phys/n:.4f}"
        if use_ssl: log_msg += f" | SSL: {total_ssl/n:.4f}"
        print(log_msg)
    print("Stage II finished.\n")

# ==================== 评估 ====================
def find_best_threshold(model, val_loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, _ = model(x, stage='finetune')
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(y.numpy())
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx]

def evaluate(model, test_loader, threshold, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits, _ = model(x, stage='finetune')
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(y.numpy())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    pred = (y_prob > threshold).astype(int)
    return {
        'acc': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob)
    }

# ==================== 辅助 ====================
def sample_k_percent(dataset, ratio):
    idx = np.random.choice(len(dataset), int(len(dataset)*ratio), replace=False)
    return torch.utils.data.Subset(dataset, idx)

def get_subject_files():
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.txt")))
    subjects = {}
    for f in files:
        sub = os.path.basename(f)[:3]
        subjects.setdefault(sub, []).append(f)
    return subjects

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 保证卷积等操作的确定性（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 主程序 ====================
def main():
    set_seed(42)
    subjects = get_subject_files()
    all_results = {r: [] for r in FEW_SHOT_RATIOS}

    for test_sub in subjects:
        print(f"\n{'='*20} Testing on subject: {test_sub} {'='*20}")
        train_files = []
        for s in subjects:
            if s != test_sub:
                train_files.extend(subjects[s])
        test_files = subjects[test_sub]

        train_dataset_full = DaphnetDataset(train_files, WINDOW, STRIDE)
        test_dataset = DaphnetDataset(test_files, WINDOW, STRIDE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for ratio in FEW_SHOT_RATIOS:
            print(f"\n>>> Few-shot ratio: {ratio}")
            ratio_metrics = []
            for rep in range(NUM_REPEATS):
                print(f"  Repetition {rep + 1}/{NUM_REPEATS}")
                train_subset = sample_k_percent(train_dataset_full, ratio)

                # === 关键修正：从 Few-shot 样本中划分出 20% 作为独立验证集 ===
                val_size = int(0.2 * len(train_subset))
                if val_size == 0: val_size = 1  # 保证至少有一个样本
                train_size = len(train_subset) - val_size
                train_sub_for_ft, val_sub = torch.utils.data.random_split(train_subset, [train_size, val_size])

                train_loader = DataLoader(train_sub_for_ft, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
                unlabeled_loader = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True)

                model = PhysGaitSSL(
                    in_channels=9, seq_len=WINDOW, patch_size=PATCH_SIZE,
                    d_model=128, num_joints=2  # 修改为2，完美对应物理方程的维度
                ).to(DEVICE)

                optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS, eta_min=1e-6)

                # Stage I
                pretrain_stage_I(model, unlabeled_loader, optimizer, scheduler,
                                 epochs=SSL_EPOCHS, mask_ratio=MASK_RATIO,
                                 vis_weight=VISIBLE_WEIGHT, device=DEVICE)

                # Stage II
                optimizer_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=LR_FINETUNE, weight_decay=1e-4)
                finetune_stage_II(model, train_loader, val_loader, optimizer_ft,  # 传入独立的 val_loader
                                  epochs=FT_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                                  use_ssl=(SSL_WEIGHT_IN_FT > 0),
                                  use_physics=(PHYSICS_WEIGHT > 0),
                                  mask_ratio=MASK_RATIO, device=DEVICE)

                best_thresh = find_best_threshold(model, val_loader, DEVICE)
                metrics = evaluate(model, test_loader, best_thresh, DEVICE)
                print(f"    Test metrics (threshold={best_thresh:.3f}): {metrics}")
                ratio_metrics.append(metrics)

            avg_metrics = {k: np.mean([m[k] for m in ratio_metrics]) for k in ratio_metrics[0]}
            all_results[ratio].append(avg_metrics)
            print(f"  Average over {NUM_REPEATS} repeats: {avg_metrics}")

    print("\n" + "="*50)
    print("FINAL RESULTS (averaged across subjects and repeats)")
    for ratio in FEW_SHOT_RATIOS:
        ratio_avg = {}
        for key in all_results[ratio][0].keys():
            values = [res[key] for res in all_results[ratio]]
            ratio_avg[key] = np.mean(values)
            ratio_std = np.std(values)
            print(f"Ratio {ratio}: {key} = {ratio_avg[key]:.4f} ± {ratio_std:.4f}")

if __name__ == "__main__":
    main()