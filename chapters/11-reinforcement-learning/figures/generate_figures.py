#!/usr/bin/env python3
"""Generate figures for Chapter 11: Reinforcement Learning"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'rl': '#6F42C1', 'traditional': '#6C757D', 'state': '#4A90D9',
    'action': '#28A745', 'reward': '#FFC107', 'policy': '#E83E8C',
    'safe': '#28A745', 'unsafe': '#DC3545', 'primary': '#4A90D9'
}

def draw_box(ax, x, y, w, h, label, color, sub=None):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='white', lw=2))
    if sub:
        ax.text(x+w/2, y+h/2+0.1, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        ax.text(x+w/2, y+h/2-0.15, sub, ha='center', va='center', fontsize=6, color='white', alpha=0.9)
    else:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

def create_rl_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6, 5.7, 'RL vs Traditional Optimization', fontsize=14, fontweight='bold', ha='center')

    # Traditional
    ax.text(3, 5, 'Traditional', fontsize=11, fontweight='bold', ha='center')
    draw_box(ax, 1.5, 3.5, 3, 1.2, 'Rule-Based', COLORS['traditional'], 'If CPU>80% scale')
    draw_box(ax, 1.5, 2, 3, 1.2, 'Static Thresholds', COLORS['traditional'], 'No learning')

    # RL
    ax.text(9, 5, 'Reinforcement Learning', fontsize=11, fontweight='bold', ha='center')
    draw_box(ax, 7.5, 3.5, 3, 1.2, 'Learned Policy', COLORS['rl'], 'Adaptive')
    draw_box(ax, 7.5, 2, 3, 1.2, 'Continuous Learning', COLORS['rl'], 'Improves over time')

    plt.tight_layout()
    plt.savefig('fig_11_1_rl_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_rl_components():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(6, 7.7, 'RL Components for Infrastructure', fontsize=14, fontweight='bold', ha='center')

    draw_box(ax, 0.5, 4.5, 3, 2.5, 'State\n\nCPU, Memory\nLatency, Load\nTime features', COLORS['state'])
    draw_box(ax, 4.5, 4.5, 3, 2.5, 'Policy\n\nNeural Network\nAction Selection', COLORS['policy'])
    draw_box(ax, 8.5, 4.5, 3, 2.5, 'Action\n\nScale Up/Down\nConfig Changes', COLORS['action'])
    draw_box(ax, 4.5, 1.5, 3, 2, 'Reward\n\nCost + Perf\n+ Reliability', COLORS['reward'])

    # Arrows
    for x1, x2 in [(3.6, 4.4), (7.6, 8.4)]:
        ax.annotate('', xy=(x2, 5.75), xytext=(x1, 5.75), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6, 4.4), xytext=(6, 3.6), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_11_2_rl_components.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_ppo_training():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(7, 5.7, 'PPO Training Loop', fontsize=14, fontweight='bold', ha='center')

    steps = [('Collect\nTrajectories', 1), ('Compute\nAdvantages', 4), ('Clip\nObjective', 7),
             ('Update\nPolicy', 10), ('Repeat', 13)]
    for label, x in steps:
        draw_box(ax, x-1, 2.5, 2, 2, label, COLORS['primary'])

    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][1]-1.1, 3.5), xytext=(steps[i][1]+1.1, 3.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_11_3_ppo_training.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_safe_deployment():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6, 5.7, 'Safe RL Deployment', fontsize=14, fontweight='bold', ha='center')

    draw_box(ax, 0.5, 2.5, 2.5, 2, 'Safety\nConstraints', COLORS['safe'])
    draw_box(ax, 3.5, 2.5, 2.5, 2, 'Gradual\nRollout', COLORS['primary'])
    draw_box(ax, 6.5, 2.5, 2.5, 2, 'A/B Testing', COLORS['policy'])
    draw_box(ax, 9.5, 2.5, 2, 2, 'Monitor &\nRollback', COLORS['reward'])

    for x in [3.1, 6.1, 9.1]:
        ax.annotate('', xy=(x, 3.5), xytext=(x-0.5, 3.5), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_11_4_safe_deployment.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 11 figures...")
    create_rl_comparison(); create_rl_components(); create_ppo_training(); create_safe_deployment()
    print("Done!")

if __name__ == "__main__":
    main()
