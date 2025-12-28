#!/usr/bin/env python3
"""Generate figures for Chapter 19: Implementation Roadmap"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'foundation': '#6C757D',
    'intelligence': '#17A2B8',
    'automation': '#28A745',
    'autonomy': '#6F42C1',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'warning': '#FFC107',
    'success': '#28A745'
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


def create_maturity_model():
    """Figure 19.1: Maturity Assessment Model"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Infrastructure Optimization Maturity Model',
            fontsize=14, fontweight='bold', ha='center')

    # Maturity levels as stacked boxes
    levels = [
        ('Level 0: Manual', 'No automation\nReactive only', COLORS['dark'], 1),
        ('Level 1: Scripted', 'Basic scripts\nScheduled tasks', COLORS['foundation'], 2),
        ('Level 2: Automated', 'CI/CD pipelines\nPolicy enforcement', COLORS['intelligence'], 3),
        ('Level 3: Intelligent', 'ML predictions\nProactive alerts', COLORS['automation'], 4),
        ('Level 4: Autonomous', 'Self-optimizing\nSelf-healing', COLORS['autonomy'], 5)
    ]

    for i, (name, desc, color, level) in enumerate(levels):
        width = 2 + level * 0.5
        x = 7 - width/2
        y = 0.8 + i * 1.3
        draw_box(ax, x, y, width, 1.1, name, color, desc)

    # Arrow showing progression
    ax.annotate('', xy=(12, 6.5), xytext=(12, 1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=3))
    ax.text(12.5, 3.75, 'Maturity\nProgression', fontsize=9, ha='left', va='center')

    plt.tight_layout()
    plt.savefig('fig_19_1_maturity_model.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_implementation_phases():
    """Figure 19.2: Four-Phase Implementation Roadmap"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Four-Phase Implementation Roadmap',
            fontsize=14, fontweight='bold', ha='center')

    # Phases
    phases = [
        ('Phase 1\nFoundation', 'Observability\nData pipelines\nBasic automation', COLORS['foundation'], 0.5),
        ('Phase 2\nIntelligence', 'ML models\nPredictive alerts\nCost analysis', COLORS['intelligence'], 3.9),
        ('Phase 3\nAutomation', 'Auto-remediation\nPolicy enforcement\nOptimization', COLORS['automation'], 7.3),
        ('Phase 4\nAutonomy', 'Self-healing\nSelf-optimizing\nMinimal intervention', COLORS['autonomy'], 10.7)
    ]

    for name, desc, color, x in phases:
        draw_box(ax, x, 3.5, 3, 3.5, f'{name}\n\n{desc}', color)

    # Arrows between phases
    for i in range(3):
        ax.annotate('', xy=(phases[i+1][3] - 0.1, 5.25),
                    xytext=(phases[i][3] + 3.1, 5.25),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Timeline bar
    draw_box(ax, 0.5, 1, 13, 1.5, 'Timeline: Foundation → Quick Wins → Scale → Optimize', COLORS['dark'])

    plt.tight_layout()
    plt.savefig('fig_19_2_implementation_phases.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_team_structure():
    """Figure 19.3: Team Structure for AIOps"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'AIOps Team Structure',
            fontsize=14, fontweight='bold', ha='center')

    # Leadership
    draw_box(ax, 5.5, 6, 3, 1.2, 'Platform Lead', COLORS['dark'])

    # Core roles
    ax.text(3.5, 5.2, 'Engineering', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.5, 3.5, 2.5, 1.3, 'ML Engineers', COLORS['primary'])
    draw_box(ax, 3.3, 3.5, 2.5, 1.3, 'Platform Eng', COLORS['primary'])

    ax.text(7, 5.2, 'Operations', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 6, 3.5, 2, 1.3, 'SRE', COLORS['automation'])

    ax.text(10.5, 5.2, 'Strategy', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 8.5, 3.5, 2, 1.3, 'FinOps', COLORS['warning'])
    draw_box(ax, 10.8, 3.5, 2.5, 1.3, 'Security', COLORS['secondary'])

    # Dotted teams
    ax.text(7, 2.3, 'Extended Team', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 2, 1, 3, 1, 'App Teams', COLORS['foundation'])
    draw_box(ax, 5.5, 1, 3, 1, 'Data Science', COLORS['foundation'])
    draw_box(ax, 9, 1, 3, 1, 'Compliance', COLORS['foundation'])

    # Connecting lines
    for x in [1.75, 4.55, 7, 9.5, 12.05]:
        ax.plot([7, x], [5.9, 4.9], 'gray', lw=1, ls='--')

    plt.tight_layout()
    plt.savefig('fig_19_3_team_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_kpi_framework():
    """Figure 19.4: KPI Framework"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'KPI Framework for Infrastructure Optimization',
            fontsize=14, fontweight='bold', ha='center')

    # KPI categories
    categories = [
        ('Operational', 'MTTR\nIncident rate\nAutomation %\nAlert noise', COLORS['primary'], 0.5),
        ('Financial', 'Cost/unit\nSavings rate\nROI\nBudget variance', COLORS['success'], 3.7),
        ('Quality', 'SLA compliance\nError rate\nDrift detection\nModel accuracy', COLORS['secondary'], 6.9),
        ('Team', 'Toil reduction\nVelocity\nCoverage\nAdoption rate', COLORS['warning'], 10.1)
    ]

    for name, metrics, color, x in categories:
        draw_box(ax, x, 3, 3, 4, f'{name}\nKPIs\n\n{metrics}', color)

    # Dashboard
    draw_box(ax, 2, 0.5, 10, 1.5, 'Executive Dashboard: Real-time visibility | Trend analysis | Benchmarking', COLORS['dark'])

    plt.tight_layout()
    plt.savefig('fig_19_4_kpi_framework.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 19 figures...")
    create_maturity_model()
    print("  Created fig_19_1_maturity_model.png")
    create_implementation_phases()
    print("  Created fig_19_2_implementation_phases.png")
    create_team_structure()
    print("  Created fig_19_3_team_structure.png")
    create_kpi_framework()
    print("  Created fig_19_4_kpi_framework.png")
    print("Done!")


if __name__ == "__main__":
    main()
