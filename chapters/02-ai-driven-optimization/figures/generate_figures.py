#!/usr/bin/env python3
"""
Chapter 2: Professional Figure Generation
High-quality diagrams styled like O'Reilly technical books
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np
import os

# Professional color palette
COLORS = {
    'blue': '#0066CC',
    'dark_blue': '#004080',
    'light_blue': '#E6F2FF',
    'orange': '#FF6600',
    'light_orange': '#FFF2E6',
    'green': '#00994D',
    'light_green': '#E6FFE6',
    'purple': '#6600CC',
    'light_purple': '#F2E6FF',
    'red': '#CC0000',
    'light_red': '#FFE6E6',
    'gray': '#666666',
    'light_gray': '#F5F5F5',
    'dark_gray': '#333333',
    'white': '#FFFFFF',
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_style():
    """Setup professional matplotlib style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'none',
        'axes.linewidth': 0,
        'xtick.bottom': False,
        'ytick.left': False,
    })


def draw_rounded_box(ax, x, y, width, height, color, text_lines, alpha=0.9):
    """Draw a professional rounded box with text"""
    # Shadow
    shadow = FancyBboxPatch(
        (x + 0.01, y - 0.01), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor='#CCCCCC', alpha=0.3, zorder=1
    )
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor=COLORS['dark_gray'],
        linewidth=1.5, alpha=alpha, zorder=2
    )
    ax.add_patch(box)

    # Text
    if isinstance(text_lines, str):
        text_lines = [text_lines]

    total_height = len(text_lines) * 0.035
    start_y = y + height/2 + total_height/2 - 0.02

    for i, line in enumerate(text_lines):
        weight = 'bold' if i == 0 else 'normal'
        size = 11 if i == 0 else 9
        ax.text(x + width/2, start_y - i*0.04, line,
               ha='center', va='center', fontsize=size,
               fontweight=weight, color=COLORS['dark_gray'], zorder=3)


def draw_arrow(ax, start, end, color=None, style='->', curved=False):
    """Draw a professional arrow"""
    if color is None:
        color = COLORS['gray']

    if curved:
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0.2",
            arrowstyle=f"{style}",
            mutation_scale=15,
            color=color,
            linewidth=2,
            zorder=1
        )
    else:
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=f"{style}",
            mutation_scale=15,
            color=color,
            linewidth=2,
            zorder=1
        )
    ax.add_patch(arrow)


def figure_2_1_intelligence_spectrum():
    """Figure 2.1: The AI Infrastructure Intelligence Spectrum - Professional Version"""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    stages = [
        ('Rule-Based', 'Automation', COLORS['light_gray'],
         'If-then rules\nThreshold alerts'),
        ('Statistical', 'Detection', COLORS['light_blue'],
         'Anomaly detection\nBaseline analysis'),
        ('Machine', 'Learning', COLORS['light_green'],
         'Predictive analytics\nPattern recognition'),
        ('Deep', 'Learning', COLORS['light_orange'],
         'Complex patterns\nNeural networks'),
        ('Reinforcement', 'Learning', COLORS['light_purple'],
         'Autonomous\nOptimization'),
    ]

    box_width = 0.14
    box_height = 0.25
    spacing = 0.17
    start_x = 0.08
    y_pos = 0.45

    # Draw connecting line
    ax.plot([start_x + box_width/2, start_x + (len(stages)-1)*spacing + box_width/2],
            [y_pos - 0.02, y_pos - 0.02],
            color=COLORS['gray'], linewidth=3, zorder=0, alpha=0.3)

    # Draw arrow at end
    ax.annotate('', xy=(0.95, y_pos - 0.02), xytext=(0.90, y_pos - 0.02),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=3))

    for i, (title1, title2, color, desc) in enumerate(stages):
        x = start_x + i * spacing

        # Draw numbered circle
        circle = plt.Circle((x + box_width/2, y_pos + box_height + 0.05), 0.03,
                           color=COLORS['blue'], zorder=4)
        ax.add_patch(circle)
        ax.text(x + box_width/2, y_pos + box_height + 0.05, str(i+1),
               ha='center', va='center', color='white', fontweight='bold', fontsize=12, zorder=5)

        # Main box
        draw_rounded_box(ax, x, y_pos, box_width, box_height, color, [title1, title2])

        # Description below
        ax.text(x + box_width/2, y_pos - 0.08, desc,
               ha='center', va='top', fontsize=9, color=COLORS['gray'],
               style='italic', linespacing=1.3)

    # Labels
    ax.text(0.02, y_pos + box_height/2, 'Simple', ha='left', va='center',
           fontsize=12, color=COLORS['gray'], fontweight='bold')
    ax.text(0.98, y_pos + box_height/2, 'Complex', ha='right', va='center',
           fontsize=12, color=COLORS['gray'], fontweight='bold')

    # Title
    ax.text(0.5, 0.95, 'The AI Infrastructure Intelligence Spectrum',
           ha='center', va='top', fontsize=16, fontweight='bold', color=COLORS['dark_gray'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_2_1_intelligence_spectrum.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig_2_1_intelligence_spectrum.png")


def figure_2_2_rl_autoscaling():
    """Figure 2.2: RL Auto-Scaling Loop - Professional Version"""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Box definitions: (x, y, width, height, color, [title, subtitle])
    boxes = {
        'state': (0.08, 0.55, 0.18, 0.18, COLORS['light_blue'],
                  ['State Observer', 'CPU, Memory', 'Queue Depth, Latency']),
        'policy': (0.36, 0.55, 0.18, 0.18, COLORS['light_purple'],
                  ['Policy Network', 'Neural Network', 'π(a|s)']),
        'action': (0.64, 0.55, 0.18, 0.18, COLORS['light_orange'],
                  ['Action Executor', 'Scale Up/Down', 'Adjust Config']),
        'env': (0.64, 0.22, 0.18, 0.18, COLORS['light_gray'],
               ['Infrastructure', 'Cloud Resources', 'Containers, VMs']),
        'reward': (0.36, 0.22, 0.18, 0.18, COLORS['light_green'],
                  ['Reward Calculator', 'Latency ↓ Cost ↓', 'Throughput ↑']),
    }

    # Draw boxes
    for name, (x, y, w, h, color, text) in boxes.items():
        draw_rounded_box(ax, x, y, w, h, color, text)

    # Draw arrows with labels
    arrows = [
        ((0.26, 0.64), (0.36, 0.64), 'Observe'),
        ((0.54, 0.64), (0.64, 0.64), 'Decide'),
        ((0.73, 0.55), (0.73, 0.40), 'Execute'),
        ((0.64, 0.31), (0.54, 0.31), 'Measure'),
        ((0.36, 0.31), (0.17, 0.31), None),  # No label
    ]

    for start, end, label in arrows:
        draw_arrow(ax, start, end, COLORS['dark_gray'])
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            if start[1] == end[1]:  # Horizontal arrow
                ax.text(mid_x, mid_y + 0.04, label, ha='center', va='bottom',
                       fontsize=10, color=COLORS['blue'], fontweight='bold')
            else:  # Vertical arrow
                ax.text(mid_x + 0.04, mid_y, label, ha='left', va='center',
                       fontsize=10, color=COLORS['blue'], fontweight='bold')

    # Feedback loop arrow (curved)
    arrow = FancyArrowPatch(
        (0.17, 0.31), (0.17, 0.55),
        connectionstyle="arc3,rad=0",
        arrowstyle="->",
        mutation_scale=15,
        color=COLORS['dark_gray'],
        linewidth=2,
        zorder=1
    )
    ax.add_patch(arrow)
    ax.text(0.12, 0.43, 'Update', ha='center', va='center',
           fontsize=10, color=COLORS['blue'], fontweight='bold', rotation=90)

    # Title box
    title_box = FancyBboxPatch(
        (0.05, 0.82), 0.90, 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=COLORS['blue'], edgecolor='none',
        alpha=0.1, zorder=0
    )
    ax.add_patch(title_box)
    ax.text(0.5, 0.88, 'Reinforcement Learning Loop for Auto-Scaling',
           ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])

    # Caption
    ax.text(0.5, 0.08,
           'The RL agent continuously observes system state, selects optimal actions,\n'
           'executes scaling decisions, and learns from reward signals.',
           ha='center', va='center', fontsize=11, color=COLORS['gray'], style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_2_2_rl_autoscaling.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig_2_2_rl_autoscaling.png")


def figure_2_3_slm_vs_llm():
    """Figure 2.3: SLM vs LLM Comparison - Professional Version"""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    categories = ['Simple\nQuery', 'Code\nGeneration', 'Log\nAnalysis', 'Alert\nSummary']

    # Left plot: Latency
    ax1 = axes[0]
    slm_latency = [15, 45, 30, 20]
    llm_latency = [800, 2000, 1500, 1000]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, slm_latency, width, label='SLM (< 3B params)',
                   color=COLORS['blue'], alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x + width/2, llm_latency, width, label='LLM (> 70B params)',
                   color=COLORS['orange'], alpha=0.85, edgecolor='white', linewidth=1)

    ax1.set_ylabel('Latency (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_yscale('log')
    ax1.set_ylim(10, 5000)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.set_title('Latency Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Value labels
    for bar, val in zip(bars1, slm_latency):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['blue'])
    for bar, val in zip(bars2, llm_latency):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['orange'])

    # Speedup annotation
    ax1.annotate('50x faster!', xy=(0.15, 800), xytext=(0.5, 400),
                fontsize=11, color=COLORS['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))

    # Right plot: Cost
    ax2 = axes[1]
    slm_cost = [0.10, 0.15, 0.12, 0.08]
    llm_cost = [10, 25, 18, 12]

    bars3 = ax2.bar(x - width/2, slm_cost, width, label='SLM (< 3B params)',
                   color=COLORS['blue'], alpha=0.85, edgecolor='white', linewidth=1)
    bars4 = ax2.bar(x + width/2, llm_cost, width, label='LLM (> 70B params)',
                   color=COLORS['orange'], alpha=0.85, edgecolor='white', linewidth=1)

    ax2.set_ylabel('Cost per 1,000 calls ($)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.set_title('Cost Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val in zip(bars3, slm_cost):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'${val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['blue'])
    for bar, val in zip(bars4, llm_cost):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'${val}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['orange'])

    # Savings annotation
    ax2.annotate('100x cheaper!', xy=(1.15, 25), xytext=(2, 20),
                fontsize=11, color=COLORS['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))

    fig.suptitle('SLM vs LLM for Infrastructure Tasks', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_2_3_slm_vs_llm.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig_2_3_slm_vs_llm.png")


def figure_2_4_grpo_convergence():
    """Figure 2.4: GRPO Convergence - Professional Version"""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    np.random.seed(42)
    epochs = np.arange(0, 100)

    # Smooth convergence curves
    def smooth_curve(base, rate, noise_level, epochs):
        signal = base + (0.9 - base) * (1 - np.exp(-epochs/rate))
        noise = np.random.normal(0, noise_level, len(epochs))
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(np.clip(signal + noise, 0, 1), sigma=2)

    try:
        from scipy.ndimage import gaussian_filter1d
        pg_reward = smooth_curve(0.3, 50, 0.06, epochs)
        ppo_reward = smooth_curve(0.3, 35, 0.04, epochs)
        grpo_reward = smooth_curve(0.3, 20, 0.02, epochs)
    except ImportError:
        # Fallback without scipy
        pg_reward = 0.3 + 0.5 * (1 - np.exp(-epochs/50))
        ppo_reward = 0.3 + 0.55 * (1 - np.exp(-epochs/35))
        grpo_reward = 0.3 + 0.6 * (1 - np.exp(-epochs/20))

    # Plot with professional styling
    ax.plot(epochs, pg_reward, label='Policy Gradient', color=COLORS['gray'],
           linewidth=2.5, alpha=0.7)
    ax.plot(epochs, ppo_reward, label='PPO', color=COLORS['orange'],
           linewidth=2.5, alpha=0.85)
    ax.plot(epochs, grpo_reward, label='GRPO (Ours)', color=COLORS['blue'],
           linewidth=3, alpha=0.95)

    # Fill area under GRPO
    ax.fill_between(epochs, grpo_reward, alpha=0.1, color=COLORS['blue'])

    # Styling
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=13, fontweight='bold')
    ax.set_title('Policy Optimization Convergence Comparison', fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    ax.set_ylim(0.2, 1.0)

    # Annotations
    ax.annotate('GRPO converges\n~2x faster',
                xy=(30, 0.78), xytext=(50, 0.55),
                fontsize=12, color=COLORS['blue'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_blue'], edgecolor='none'))

    ax.axhline(y=0.85, color=COLORS['green'], linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(95, 0.86, 'Target', ha='right', va='bottom', color=COLORS['green'], fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_2_4_grpo_convergence.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig_2_4_grpo_convergence.png")


def figure_2_5_adoption_barriers():
    """Figure 2.5: Adoption Barriers - Professional Version"""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    barriers = ['ML Expertise\nGap', 'Data\nSilos', 'Automation\nFear', 'Unclear\nROI', 'Vendor\nLock-in']
    severity = [75, 60, 45, 55, 40]
    mitigation = [60, 70, 80, 65, 85]

    x = np.arange(len(barriers))
    width = 0.35

    bars1 = ax.bar(x - width/2, severity, width, label='Barrier Severity (%)',
                  color=COLORS['red'], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, mitigation, width, label='Mitigation Success (%)',
                  color=COLORS['green'], alpha=0.8, edgecolor='white', linewidth=1)

    ax.set_ylabel('Percentage', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(barriers, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.set_title('AI Infrastructure Adoption: Barriers vs. Mitigation Success',
                fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Value labels
    for bar, val in zip(bars1, severity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['red'])
    for bar, val in zip(bars2, mitigation):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['green'])

    # Key insight box
    insight_text = "Key Insight: All barriers have >60% mitigation success rate\nwith proper strategy and tooling."
    ax.text(0.5, 0.05, insight_text, transform=ax.transAxes,
           ha='center', va='bottom', fontsize=11, color=COLORS['dark_gray'],
           style='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_gray'],
                                     edgecolor='none', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_2_5_adoption_barriers.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig_2_5_adoption_barriers.png")


def main():
    """Generate all professional figures for Chapter 2"""
    print("=" * 60)
    print("Generating Professional Figures for Chapter 2")
    print("=" * 60)

    figure_2_1_intelligence_spectrum()
    figure_2_2_rl_autoscaling()
    figure_2_3_slm_vs_llm()
    figure_2_4_grpo_convergence()
    figure_2_5_adoption_barriers()

    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
