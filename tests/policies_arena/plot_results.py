"""
Policy Arena Results Visualization

Generates a 2x2 figure with survival rate progression and velocity tracking error heatmaps.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# USER CONFIG - Edit this timestamp to match your experiment
# ============================================================
TIMESTAMP = "2026-01-31-10-58-32"

# ============================================================
# Path Construction
# ============================================================
EXPERIMENT_DIR = os.path.join("exps", "T1", TIMESTAMP)
CSV_PATH = os.path.join(EXPERIMENT_DIR, "results.csv")
OUTPUT_PATH = os.path.join(EXPERIMENT_DIR, "results_plot.png")


def load_results(csv_path):
    """Load results CSV into DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_survival_rate(ax, df):
    """Plot survival rate progression (line plot)."""
    policies = df['policy_name'].unique()
    
    for policy in policies:
        policy_data = df[df['policy_name'] == policy].sort_values('stage')
        ax.plot(policy_data['stage'], policy_data['survival_rate'], 
                marker='o', label=policy, linewidth=2, markersize=6)
    
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Pass threshold (90%)')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Survival Rate')
    ax.set_title('Survival Rate Progression')
    
    y_min = df['survival_rate'].min()
    y_max = df['survival_rate'].max()
    padding = (y_max - y_min) * 0.15
    ax.set_ylim(max(0, y_min - padding), min(1.05, y_max + padding))
    
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    stages = df['stage'].unique()
    ax.set_xticks(stages)


def plot_error_heatmap(ax, df, error_column, title):
    """Plot velocity error heatmap."""
    policies = df['policy_name'].unique()
    stages = sorted(df['stage'].unique())
    
    pivot = df.pivot(index='policy_name', columns='stage', values=error_column)
    pivot = pivot.reindex(index=policies, columns=stages)
    
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)
    
    ax.set_xlabel('Stage')
    ax.set_ylabel('Policy')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    for i in range(len(policies)):
        for j in range(len(stages)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > pivot.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=8, color=text_color)


def main():
    print(f"Loading results from: {CSV_PATH}")
    df = load_results(CSV_PATH)
    
    print(f"Found {len(df['policy_name'].unique())} policies, {len(df['stage'].unique())} stages")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Policy Arena Results — {TIMESTAMP}', fontsize=14, fontweight='bold')
    
    plot_survival_rate(axes[0, 0], df)
    plot_error_heatmap(axes[0, 1], df, 'vx_error', 'Vx Tracking Error')
    plot_error_heatmap(axes[1, 0], df, 'vy_error', 'Vy Tracking Error')
    plot_error_heatmap(axes[1, 1], df, 'vyaw_error', 'Vyaw Tracking Error')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()