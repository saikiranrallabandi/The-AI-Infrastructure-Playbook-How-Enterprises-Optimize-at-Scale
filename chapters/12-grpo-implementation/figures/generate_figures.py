#!/usr/bin/env python3
"""Generate figures for Chapter 12: GRPO Implementation Deep Dive"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'grpo': '#6F42C1',
    'state': '#4A90D9',
    'action': '#28A745',
    'reward': '#FFC107',
    'policy': '#E83E8C',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'safe': '#28A745',
    'group': '#17A2B8'
}


def draw_box(ax, x, y, w, h, label, color, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='white', lw=2))
    if sub:
        ax.text(x + w/2, y + h/2 + 0.12, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.12, sub, ha='center', va='center',
                fontsize=6, color='white', alpha=0.9)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')


def create_grpo_architecture():
    """Figure 12.1: GRPO Architecture Overview"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'GRPO Architecture for Infrastructure Optimization',
            fontsize=14, fontweight='bold', ha='center')

    # State Encoder
    ax.text(1.5, 7, 'State Encoding', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5.5, 2.4, 1.2, 'Metrics History', COLORS['state'], 'LSTM Encoder')
    draw_box(ax, 0.3, 4, 2.4, 1.2, 'Current State', COLORS['state'], 'Normalizer')

    # Policy Network
    ax.text(5, 7, 'Policy Network', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.5, 4.5, 3, 2.5, 'Transformer\n\nMulti-head\nAttention', COLORS['policy'])

    # Group Sampling
    ax.text(9, 7, 'Group Sampling', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 7.5, 5.5, 3, 1.2, 'Sample K Actions', COLORS['group'], 'Per state')
    draw_box(ax, 7.5, 4, 3, 1.2, 'Execute & Observe', COLORS['action'], 'Get rewards')

    # Advantage Computation
    ax.text(12.5, 7, 'GRPO Update', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 11.3, 5.5, 2.4, 1.2, 'Group-Relative', COLORS['grpo'], 'Advantages')
    draw_box(ax, 11.3, 4, 2.4, 1.2, 'Policy Update', COLORS['grpo'], 'Clipped obj')

    # Bottom: Reward Components
    ax.text(7, 2.8, 'Multi-Objective Reward', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 2, 1.2, 2.2, 1.2, 'Cost', COLORS['reward'], '30%')
    draw_box(ax, 4.5, 1.2, 2.2, 1.2, 'Performance', COLORS['reward'], '30%')
    draw_box(ax, 7, 1.2, 2.2, 1.2, 'Reliability', COLORS['reward'], '25%')
    draw_box(ax, 9.5, 1.2, 2.2, 1.2, 'Efficiency', COLORS['reward'], '15%')

    # Arrows
    ax.annotate('', xy=(3.4, 5.5), xytext=(2.8, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(7.4, 5.5), xytext=(6.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(11.2, 5.5), xytext=(10.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Reward to group
    ax.annotate('', xy=(9, 3.9), xytext=(9, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_12_1_grpo_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_advantage_comparison():
    """Figure 12.2: Traditional vs GRPO Advantage Estimation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Traditional advantages
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('Traditional (GAE)', fontsize=12, fontweight='bold')

    draw_box(ax1, 1, 5.5, 2.5, 1.5, 'Rewards', COLORS['reward'])
    draw_box(ax1, 4.5, 5.5, 2.5, 1.5, 'Value Function', COLORS['policy'], 'Learned V(s)')
    draw_box(ax1, 2.5, 2.5, 4, 2, 'A(s,a) = R + γV(s\') - V(s)\n\nEstimation Error', COLORS['secondary'])

    ax1.annotate('', xy=(3.5, 5.4), xytext=(2.5, 5.4),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.annotate('', xy=(4.5, 4.6), xytext=(4.5, 5.4),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # GRPO advantages
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('GRPO (Group-Relative)', fontsize=12, fontweight='bold')

    # Group of rewards
    for i, offset in enumerate([0, 2.2, 4.4, 6.6]):
        draw_box(ax2, 1 + offset * 0.3, 5.5, 1.8, 1.2, f'R_{i+1}', COLORS['reward'])

    draw_box(ax2, 2.5, 2.5, 4, 2, 'A_i = (R_i - μ_group) / σ_group\n\nNo V(s) needed!', COLORS['grpo'])

    ax2.annotate('', xy=(4.5, 4.6), xytext=(4.5, 5.4),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_12_2_advantage_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_action_space():
    """Figure 12.3: Infrastructure Action Space Design"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Hierarchical Infrastructure Action Space',
            fontsize=14, fontweight='bold', ha='center')

    # Scaling Actions
    ax.text(2.5, 6.8, 'Scaling Actions', fontsize=10, fontweight='bold', ha='center')
    scaling = ['--4', '--2', '-1', '0', '+1', '+2', '+4']
    for i, s in enumerate(scaling):
        color = COLORS['safe'] if s == '0' else COLORS['action']
        draw_box(ax, 0.3 + i * 0.7, 5.8, 0.6, 0.8, s, color)

    # Config Actions
    ax.text(2.5, 5, 'Config Actions', fontsize=10, fontweight='bold', ha='center')
    configs = ['Batch-', 'Batch+', 'Timeout-', 'Timeout+', 'Concur-', 'Concur+', 'None']
    for i, c in enumerate(configs):
        draw_box(ax, 0.3 + i * 0.7, 4, 0.6, 0.8, c[:4], COLORS['secondary'])

    # Traffic Actions
    ax.text(2.5, 3.2, 'Traffic Weight', fontsize=10, fontweight='bold', ha='center')
    for i in range(11):
        draw_box(ax, 0.3 + i * 0.45, 2.2, 0.4, 0.8, f'{i*10}%', COLORS['primary'])

    # Action Composition
    ax.text(10, 6.8, 'Action Composition', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 7, 5, 6, 2.5,
             'Total Actions = |Scaling| × |Config| × |Traffic|\n'
             '             = 7 × 7 × 11 = 539\n\n'
             'Action = (scale +2, batch+, 70%)', COLORS['dark'])

    # Decoder
    draw_box(ax, 7.5, 1.5, 2, 1.5, 'Action\nDecoder', COLORS['policy'])
    draw_box(ax, 10.5, 1.5, 2.5, 1.5, 'K8s API\nCommands', COLORS['action'])

    ax.annotate('', xy=(10.4, 2.25), xytext=(9.6, 2.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_12_3_action_space.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_training_pipeline():
    """Figure 12.4: GRPO Training Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'GRPO Training Pipeline',
            fontsize=14, fontweight='bold', ha='center')

    # Collection Phase
    ax.text(2, 6.5, 'Collection', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.5, 4.5, 3, 1.8, 'Sample Groups\n\nK actions per state', COLORS['group'])

    # Environment
    draw_box(ax, 0.5, 2.2, 3, 1.8, 'Infrastructure\nEnvironment', COLORS['state'])

    # Processing Phase
    ax.text(6, 6.5, 'Processing', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 4, 5, 1.8, 1.2, 'Compute\nReturns', COLORS['reward'])
    draw_box(ax, 6, 5, 2, 1.2, 'Group-Rel\nAdvantages', COLORS['grpo'])

    # Update Phase
    ax.text(10.5, 6.5, 'Update', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 8.5, 4.5, 2, 1.8, 'Clipped\nObjective', COLORS['policy'])
    draw_box(ax, 11, 4.5, 2.5, 1.8, 'AdamW\nOptimizer', COLORS['secondary'])

    # Safety Layer
    ax.text(10.5, 2.5, 'Safety Layer', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 8.5, 1, 2, 1.2, 'Constraint\nCheck', COLORS['safe'])
    draw_box(ax, 11, 1, 2.5, 1.2, 'Gradual\nRollout', COLORS['safe'])

    # Arrows
    ax.annotate('', xy=(3.9, 5.5), xytext=(3.6, 5.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(5.9, 5.5), xytext=(5.9, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(8.4, 5.3), xytext=(8.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(10.9, 5.3), xytext=(10.6, 5.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Loop back
    ax.annotate('', xy=(2, 4.4), xytext=(2, 4.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(2, 2.1), xytext=(2, 4.4),
                arrowprops=dict(arrowstyle='<-', color='gray', lw=2,
                               connectionstyle='arc3,rad=0.3'))

    plt.tight_layout()
    plt.savefig('fig_12_4_training_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 12 figures...")
    create_grpo_architecture()
    print("  Created fig_12_1_grpo_architecture.png")
    create_advantage_comparison()
    print("  Created fig_12_2_advantage_comparison.png")
    create_action_space()
    print("  Created fig_12_3_action_space.png")
    create_training_pipeline()
    print("  Created fig_12_4_training_pipeline.png")
    print("Done!")


if __name__ == "__main__":
    main()
