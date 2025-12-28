#!/usr/bin/env python3
"""Generate figures for Chapter 20: Future of Autonomous Infrastructure"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'manual': '#6C757D',
    'assisted': '#17A2B8',
    'automated': '#FFC107',
    'intelligent': '#28A745',
    'autonomous': '#6F42C1',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'future': '#E83E8C'
}


def draw_box(ax, x, y, w, h, label, color, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='white', lw=2))
    if sub:
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.15, sub, ha='center', va='center',
                fontsize=6, color='white', alpha=0.9)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')


def create_future_landscape():
    """Figure 20.1: Evolution to Autonomous Infrastructure"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Evolution Toward Fully Autonomous Infrastructure',
            fontsize=14, fontweight='bold', ha='center')

    # Evolution stages as ascending path
    stages = [
        ('Manual', 'Scripts\nTickets', COLORS['manual'], 1, 1),
        ('Assisted', 'Alerts\nDashboards', COLORS['assisted'], 3.5, 2),
        ('Automated', 'CI/CD\nAuto-scale', COLORS['automated'], 6, 3),
        ('Intelligent', 'ML Predictions\nAuto-remediate', COLORS['intelligent'], 8.5, 4.5),
        ('Autonomous', 'Self-healing\nSelf-evolving', COLORS['autonomous'], 11, 6)
    ]

    for name, desc, color, x, y in stages:
        draw_box(ax, x, y, 2.3, 1.5, f'{name}\n\n{desc}', color)

    # Connecting arrows (ascending)
    for i in range(len(stages)-1):
        x1, y1 = stages[i][3] + 2.3, stages[i][4] + 0.75
        x2, y2 = stages[i+1][3], stages[i+1][4] + 0.75
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Labels
    ax.text(1, 7, 'Past', fontsize=10, color='gray')
    ax.text(6, 7, 'Present', fontsize=10, color='gray')
    ax.text(11, 7, 'Future', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('fig_20_1_future_landscape.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_emerging_tech():
    """Figure 20.2: Emerging AI Technologies"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Emerging AI Technologies for Infrastructure',
            fontsize=14, fontweight='bold', ha='center')

    # Technology categories
    categories = [
        ('Foundation\nModels', 'Multi-modal\nInfra-specific\nWorld models', COLORS['primary'], 0.5),
        ('Agentic\nSystems', 'Planning\nTool use\nMemory', COLORS['secondary'], 3.5),
        ('Causal\nML', 'Root cause\nIntervention\nCounterfactual', COLORS['intelligent'], 6.5),
        ('Self-\nEvolution', 'Strategy\nevolution\nMeta-learning', COLORS['autonomous'], 9.5)
    ]

    for name, desc, color, x in categories:
        draw_box(ax, x, 3, 3, 3.5, f'{name}\n\n{desc}', color)

    # Integration layer
    draw_box(ax, 1, 0.5, 12, 1.5, 'Unified Autonomous Platform: Observe → Understand → Decide → Act → Learn', COLORS['dark'])

    plt.tight_layout()
    plt.savefig('fig_20_2_emerging_tech.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_edge_intelligence():
    """Figure 20.3: Distributed Edge Intelligence"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Distributed Edge Intelligence Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Central brain
    draw_box(ax, 5.5, 5, 3, 2, 'Central\nOrchestrator\n\nGlobal model\nAggregation', COLORS['primary'])

    # Edge nodes
    edge_positions = [(1, 2), (4, 2), (7, 2), (10, 2)]
    for i, (x, y) in enumerate(edge_positions):
        draw_box(ax, x, y, 2.5, 2, f'Edge {i+1}\n\nLocal model\nLocal decision', COLORS['secondary'])

    # Connections
    for x, y in edge_positions:
        ax.plot([x + 1.25, 7], [y + 2, 5], 'gray', lw=1.5, ls='--')

    # Federated learning indicator
    ax.text(7, 1, 'Federated Learning: Models trained locally, aggregated centrally',
            fontsize=9, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('fig_20_3_edge_intelligence.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_governance():
    """Figure 20.4: Autonomous Infrastructure Governance"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Autonomous Infrastructure Governance Framework',
            fontsize=14, fontweight='bold', ha='center')

    # Governance layers
    layers = [
        ('Ethical Constraints', 'Fairness | Safety | Transparency | Accountability', COLORS['autonomous'], 6.5),
        ('Policy Framework', 'Compliance | Security | Cost | Performance', COLORS['primary'], 5),
        ('Oversight Levels', 'Logging | Notification | Approval | Blocking', COLORS['secondary'], 3.5),
        ('Audit Trail', 'Decisions | Outcomes | Explanations | Learning', COLORS['dark'], 2)
    ]

    for name, desc, color, y in layers:
        draw_box(ax, 1, y, 12, 1.2, f'{name}: {desc}', color)

    # Arrows showing flow
    for i in range(len(layers)-1):
        y1 = layers[i][3]
        y2 = layers[i+1][3] + 1.2
        ax.annotate('', xy=(7, y2), xytext=(7, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_20_4_governance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 20 figures...")
    create_future_landscape()
    print("  Created fig_20_1_future_landscape.png")
    create_emerging_tech()
    print("  Created fig_20_2_emerging_tech.png")
    create_edge_intelligence()
    print("  Created fig_20_3_edge_intelligence.png")
    create_governance()
    print("  Created fig_20_4_governance.png")
    print("Done!")


if __name__ == "__main__":
    main()
