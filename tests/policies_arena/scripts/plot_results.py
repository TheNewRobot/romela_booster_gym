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
TIMESTAMP = "2026-02-04-15-18-10"

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


def plot_terrain_breakdown(ax, experiment_dir):
    """Plot terrain breakdown as grouped bar chart."""
    summary_path = os.path.join(experiment_dir, "terrain_summary.csv")
    if not os.path.exists(summary_path):
        ax.text(0.5, 0.5, 'No terrain summary data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df_summary = pd.read_csv(summary_path)
    policies = df_summary['policy_name'].tolist()
    
    # Find available terrain groups
    terrain_cols = [c for c in df_summary.columns if c.endswith('_avg_error') and c != 'overall_avg_error']
    terrain_names = [c.replace('_avg_error', '').capitalize() for c in terrain_cols]
    
    x = np.arange(len(policies))
    n_groups = len(terrain_cols) + 1  # +1 for overall
    bar_width = 0.8 / n_groups
    
    colors = {'flat': '#4CAF50', 'slope': '#FF9800', 'rough': '#F44336', 'overall': '#2196F3'}
    
    for i, (col, name) in enumerate(zip(terrain_cols, terrain_names)):
        values = df_summary[col].fillna(0).values
        color = colors.get(name.lower(), '#9E9E9E')
        ax.bar(x + i * bar_width, values, bar_width, label=name, color=color, alpha=0.85)
    
    # Overall bar
    overall = df_summary['overall_avg_error'].fillna(0).values
    ax.bar(x + len(terrain_cols) * bar_width, overall, bar_width, label='Overall', color=colors['overall'], alpha=0.85)
    
    ax.set_xlabel('Policy')
    ax.set_ylabel('Avg Tracking Error')
    ax.set_title('Tracking Error by Terrain')
    ax.set_xticks(x + bar_width * n_groups / 2 - bar_width / 2)
    ax.set_xticklabels(policies, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def main():
    print(f"Loading results from: {CSV_PATH}")
    df = load_results(CSV_PATH)
    
    print(f"Found {len(df['policy_name'].unique())} policies, {len(df['stage'].unique())} stages")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Policy Arena Results — {TIMESTAMP}', fontsize=14, fontweight='bold')
    
    # Row 1: Survival + Per-dimension heatmaps
    plot_survival_rate(axes[0, 0], df)
    plot_error_heatmap(axes[0, 1], df, 'vx_error', 'Vx Tracking Error')
    plot_error_heatmap(axes[0, 2], df, 'vy_error', 'Vy Tracking Error')
    
    # Row 2: Vyaw heatmap + Terrain breakdown
    plot_error_heatmap(axes[1, 0], df, 'vyaw_error', 'Vyaw Tracking Error')
    plot_error_heatmap(axes[1, 1], df, 'tracking_error_full', 'Full Tracking Error')
    plot_terrain_breakdown(axes[1, 2], EXPERIMENT_DIR)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()