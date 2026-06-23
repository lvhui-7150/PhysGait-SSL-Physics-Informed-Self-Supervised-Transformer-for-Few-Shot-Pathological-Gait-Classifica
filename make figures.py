import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ==================== 配置 ====================
# 请根据需要修改为你的日志文件名
# 如果是读取新生成的补齐1.0比例的日志，请修改为 "PhysGait_Complete_Experiment_Logs.txt"
LOG_FILE = "all man.txt"
OUTPUT_DIR = "./paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 1. 解析所有受试者的最终指标 ====================
def parse_all_subjects_metrics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 按受试者分割
    subjects_raw = re.split(r'==================== Testing on subject: (\S+) ====================', text)
    all_data = {}

    for i in range(1, len(subjects_raw), 2):
        subject_id = subjects_raw[i].strip()
        subject_block = subjects_raw[i + 1]

        # 【修复1】使用 >+ 兼容 3 个或 4 个大于号 的 Few-shot ratio 分隔符
        ratio_parts = re.split(r'>+\s*Few-shot ratio:\s*([\d.]+)', subject_block)

        for j in range(1, len(ratio_parts), 2):
            ratio = float(ratio_parts[j])
            ratio_block = ratio_parts[j + 1]

            # 【修复2】先将 ratio 块按 Repetition 切分，防止跨轮次干扰
            rep_parts = re.split(r'Repetition \d+/\d+', ratio_block)

            for rep_text in rep_parts[1:]:
                # 【双格式完美兼容】同时支持单行字典格式 {'acc': ...} 和多行人类可读格式 Accuracy: ...
                acc_m = re.search(r"(?:'acc':\s*|Accuracy:\s*)([\d.]+)", rep_text)
                pre_m = re.search(r"(?:'precision':\s*|Precision:\s*)([\d.]+)", rep_text)
                rec_m = re.search(r"(?:'recall':\s*|Recall:\s*)([\d.]+)", rep_text)
                f1_m = re.search(r"(?:'f1':\s*|F1-Score:\s*)([\d.]+)", rep_text)
                auc_m = re.search(r"(?:'auc':\s*|AUC:\s*)([\d.]+)", rep_text)

                if acc_m and pre_m and rec_m and f1_m and auc_m:
                    all_data.setdefault(subject_id, []).append({
                        'ratio': ratio,
                        'acc': float(acc_m.group(1)),
                        'precision': float(pre_m.group(1)),
                        'recall': float(rec_m.group(1)),
                        'f1': float(f1_m.group(1)),
                        'auc': float(auc_m.group(1))
                    })

    # 转换为DataFrame
    df_list = []
    for sub, records in all_data.items():
        for rec in records:
            df_list.append({
                'subject': sub,
                'ratio': rec['ratio'],
                'acc': rec['acc'],
                'precision': rec['precision'],
                'recall': rec['recall'],
                'f1': rec['f1'],
                'auc': rec['auc']
            })
    return pd.DataFrame(df_list)


# ==================== 2. 提取“代表性受试者”的详细损失曲线 ====================
def get_representative_loss_data(filepath, target_subject=None):
    if not os.path.exists(filepath):
        return None, None, None, None, None

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    df_metrics = parse_all_subjects_metrics(filepath)
    if df_metrics.empty:
        return None, None, None, None, None

    if target_subject is None:
        # 计算每个受试者的平均AUC（跨所有比例），选中间值作为中位数代表
        median_sub = df_metrics.groupby('subject')['auc'].mean().sort_values().index[
            len(df_metrics.groupby('subject')) // 2]
        target_subject = median_sub
        print(f"自动选择代表性受试者: {target_subject} (中位数表现)")

    pattern = rf'==================== Testing on subject: {target_subject} ====================(.*?)(?===================== Testing on subject:|$)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"未找到受试者 {target_subject} 的数据")

    block = match.group(1)

    # 兼容 > 分隔符
    ratio_blocks = re.split(r'>+\s*Few-shot ratio:\s*([\d.]+)', block)
    for idx in range(1, len(ratio_blocks), 2):
        ratio = float(ratio_blocks[idx])
        if abs(ratio - 0.1) < 1e-4:  # 取 0.1 比例展示难度最大场景下的收敛曲线
            ratio_data = ratio_blocks[idx + 1]
            rep_parts = re.split(r'Repetition \d+/\d+', ratio_data)
            if len(rep_parts) > 1:
                rep_block = rep_parts[1]
                # 兼容灵活的空格与管道符 | 分隔
                epoch_lines = re.findall(
                    r'Epoch\s+(\d+)/\d+.*?Cls:\s*([\d.]+)\s*\|\s*Phys:\s*([\d.]+)\s*\|\s*SSL:\s*([\d.]+)', rep_block)
                if epoch_lines:
                    epochs = [int(e) for e, _, _, _ in epoch_lines]
                    cls = [float(c) for _, c, _, _ in epoch_lines]
                    phys = [float(p) for _, _, p, _ in epoch_lines]
                    ssl = [float(s) for _, _, _, s in epoch_lines]
                    return target_subject, epochs, cls, phys, ssl
            break
    return target_subject, None, None, None, None


# ==================== 3. 绘图引擎 ====================
def plot_cross_subject_figure(df_metrics):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # ---------- Figure 1: 跨受试者箱线图（证明泛化性） ----------
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图A: AUC分布
    ax1 = axes[0]
    sns.boxplot(data=df_metrics, x='ratio', y='auc', hue='ratio', palette='viridis',
                ax=ax1, linewidth=2, fliersize=6, legend=False)
    # 叠加散点（显示每个受试者的实际点）
    sns.stripplot(data=df_metrics, x='ratio', y='auc', color='black', size=4, alpha=0.3, ax=ax1)
    ax1.set_xlabel('Label Fraction (ρ)', fontsize=13)
    ax1.set_ylabel('AUC-ROC', fontsize=13)
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # ax1.set_title('(a) Cross-Subject AUC Distribution', fontweight='bold')

    # 子图B: F1分布
    ax2 = axes[1]
    sns.boxplot(data=df_metrics, x='ratio', y='f1', hue='ratio', palette='magma',
                ax=ax2, linewidth=2, fliersize=6, legend=False)
    sns.stripplot(data=df_metrics, x='ratio', y='f1', color='black', size=4, alpha=0.3, ax=ax2)
    ax2.set_xlabel('Label Fraction (ρ)', fontsize=13)
    ax2.set_ylabel('F1-Score', fontsize=13)
    # 【修改】上限调整为 1.0，完美展示 1.0 全监督对照组达到 95% 以上的高水准表现
    ax2.set_ylim(0.6, 1.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    # ax2.set_title('(b) Cross-Subject F1 Distribution', fontweight='bold')

    # fig1.suptitle('PhysGait-SSL: Generalization Performance Across 10 Subjects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Cross_Subject_Boxplot.png', dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"跨受试者箱线图已成功保存至: {OUTPUT_DIR}/Cross_Subject_Boxplot.png")

    # ---------- Figure 2: 代表性受试者损失曲线 ----------
    target_sub, epochs, cls, phys, ssl = get_representative_loss_data(LOG_FILE)

    if epochs:
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 子图A: 损失收敛（双轴）
        ax1 = axes[0]
        ax2_twin = ax1.twinx()
        l1, = ax1.plot(epochs, cls, 'o-', color='#2E86AB', label='Cls Loss', linewidth=2.5)
        l2, = ax1.plot(epochs, ssl, 's-', color='#A23B72', label='SSL Loss', linewidth=2.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cls / SSL Loss", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        l3, = ax2_twin.plot(epochs, phys, 'D-', color='#F18F01', label='Physics Loss', linewidth=2.5)
        ax2_twin.set_ylabel("Physics Loss", color='#F18F01')
        ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
        ax2_twin.spines['right'].set_color('#F18F01')

        # 标志解冻编码器的分界线
        ax1.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7)
        max_val = max(max(cls), max(ssl))
        ax1.text(3.0, max_val * 0.85, 'Frozen', ha='center', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        ax1.text(8.0, max_val * 0.85, 'Unfrozen', ha='center',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        ax1.legend([l1, l2, l3], ['Cls Loss', 'SSL Loss', 'Physics Loss'], loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.5)
        # ax1.set_title(f'(a) Loss Curves for Subject {target_sub} (ρ=0.1)', fontweight='bold')

        # 子图B: 物理损失对数收敛
        ax3 = axes[1]
        ax3.plot(epochs, phys, 'D-', color='#F18F01', linewidth=2.5, markersize=8)
        ax3.set_yscale('log')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Physics Loss (Log Scale)")
        ax3.grid(True, linestyle=':', alpha=0.6)
        # ax3.set_title(f'(b) Physics Convergence for Subject {target_sub} (ρ=0.1)', fontweight='bold')

        # fig2.suptitle('Representative Case Study: Model Training Dynamics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/Representative_Loss_Curves.png', dpi=600, bbox_inches='tight')
        plt.close(fig2)
        print(f"代表性损失曲线图已成功保存至: {OUTPUT_DIR}/Representative_Loss_Curves.png")
    else:
        print("无法解析损失曲线数据，请检查日志中是否有匹配的 Epoch 损失格式。")

    # ---------- 打印统计摘要 ----------
    print("\n===== 跨受试者统计摘要 =====")
    summary = df_metrics.groupby('ratio')[['auc', 'f1']].agg(['mean', 'std']).round(4)
    print(summary)


# ==================== 主程序 ====================
if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        print(f"错误：文件 {LOG_FILE} 未找到。请确保 LOG_FILE 变量的名字与你的日志文件名完全一致。")
    else:
        print(f"正在智能解析全量日志数据: {LOG_FILE} ...")
        df = parse_all_subjects_metrics(LOG_FILE)
        if df.empty:
            print("解析失败：未提取到任何有效指标。请检查日志格式是否符合预期。")
        else:
            print(f"成功解析 {df['subject'].nunique()} 个受试者，共 {len(df)} 条实验记录。")
            plot_cross_subject_figure(df)