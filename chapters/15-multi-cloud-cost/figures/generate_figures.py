#!/usr/bin/env python3
"""Generate figures for Chapter 15: Multi-Cloud Cost Optimization"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'aws': '#FF9900',
    'gcp': '#4285F4',
    'azure': '#0089D6',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'savings': '#28A745',
    'warning': '#FFC107',
    'cost': '#DC3545'
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


def create_multicloud_overview():
    """Figure 15.1: Multi-Cloud Cost Challenge"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Multi-Cloud Cost Optimization Challenge',
            fontsize=14, fontweight='bold', ha='center')

    # Cloud providers
    draw_box(ax, 0.5, 5, 3, 2, 'AWS', COLORS['aws'], 'EC2, S3, RDS...')
    draw_box(ax, 4, 5, 3, 2, 'GCP', COLORS['gcp'], 'GCE, GCS, BigQuery...')
    draw_box(ax, 7.5, 5, 3, 2, 'Azure', COLORS['azure'], 'VMs, Blob, SQL...')

    # Challenges
    ax.text(5.5, 4, 'Challenges', fontsize=10, fontweight='bold', ha='center')
    challenges = ['Different pricing models', 'Multiple billing systems', 'No unified view', 'Complex discounts']
    for i, c in enumerate(challenges):
        draw_box(ax, 0.5 + i * 3.2, 2.5, 3, 1, c, COLORS['cost'])

    # Solution
    draw_box(ax, 3, 0.5, 8, 1.5, 'AI-Powered Unified Cost Optimization', COLORS['savings'])

    # Arrows
    for x in [2, 5.5, 9]:
        ax.annotate('', xy=(7, 2.2), xytext=(x, 4.9),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_15_1_multicloud_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_cost_aggregation():
    """Figure 15.2: Cost Aggregation Architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Unified Cost Aggregation Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Data sources
    ax.text(1.5, 6.8, 'Cloud APIs', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5.5, 2.4, 1, 'AWS Cost Explorer', COLORS['aws'])
    draw_box(ax, 0.3, 4.2, 2.4, 1, 'GCP Billing', COLORS['gcp'])
    draw_box(ax, 0.3, 2.9, 2.4, 1, 'Azure Cost Mgmt', COLORS['azure'])

    # Connectors
    ax.text(4.5, 6.8, 'Connectors', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.3, 3.5, 2.4, 3, 'Normalized\nData Model\n\nCloudCost\nObject', COLORS['dark'])

    # Aggregator
    ax.text(8, 6.8, 'Aggregator', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 6.5, 3.5, 3, 3, 'Multi-Cloud\nAggregator\n\nCost Summary\nRecommendations', COLORS['primary'])

    # Outputs
    ax.text(12, 6.8, 'Outputs', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 10.5, 5.5, 3, 1, 'Dashboards', COLORS['secondary'])
    draw_box(ax, 10.5, 4.2, 3, 1, 'Alerts', COLORS['warning'])
    draw_box(ax, 10.5, 2.9, 3, 1, 'Automation', COLORS['savings'])

    # Arrows
    ax.annotate('', xy=(3.2, 5), xytext=(2.8, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6.4, 5), xytext=(5.8, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(10.4, 5), xytext=(9.6, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_15_2_cost_aggregation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_workload_placement():
    """Figure 15.3: Workload Placement Decision Flow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Intelligent Workload Placement',
            fontsize=14, fontweight='bold', ha='center')

    # Input
    draw_box(ax, 0.5, 4, 2.5, 2.5, 'Workload\nRequirements\n\nCPU, Memory\nGPU, Network', COLORS['primary'])

    # Analysis
    ax.text(5.5, 6.8, 'Analysis', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.5, 5, 2, 1.5, 'Cost\nComparison', COLORS['aws'])
    draw_box(ax, 5.8, 5, 2, 1.5, 'Performance\nMatch', COLORS['gcp'])
    draw_box(ax, 3.5, 3, 2, 1.5, 'Compliance\nCheck', COLORS['azure'])
    draw_box(ax, 5.8, 3, 2, 1.5, 'Data\nLocality', COLORS['secondary'])

    # Decision
    draw_box(ax, 8.5, 3.5, 2.5, 2.5, 'Placement\nOptimizer\n\nML Ranking', COLORS['dark'])

    # Outputs
    ax.text(12.5, 6.8, 'Recommendation', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 11.5, 5, 2.2, 1.2, 'Best Cloud', COLORS['savings'])
    draw_box(ax, 11.5, 3.5, 2.2, 1.2, 'Instance Type', COLORS['savings'])
    draw_box(ax, 11.5, 2, 2.2, 1.2, 'Pricing Option', COLORS['savings'])

    # Arrows
    ax.annotate('', xy=(3.4, 5.25), xytext=(3.1, 5.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(8.4, 4.75), xytext=(7.9, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(11.4, 4.5), xytext=(11.1, 4.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_15_3_workload_placement.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_finops_automation():
    """Figure 15.4: FinOps Automation Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'FinOps Automation Pipeline',
            fontsize=14, fontweight='bold', ha='center')

    # Analysis
    ax.text(2, 6.5, 'Analysis', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.5, 4.5, 3, 1.8, 'Cost Data\nAnalysis', COLORS['primary'])

    # Recommendations
    ax.text(6, 6.5, 'Recommendations', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 4, 5, 2, 1.2, 'Rightsize', COLORS['savings'])
    draw_box(ax, 6.3, 5, 2, 1.2, 'Terminate', COLORS['cost'])
    draw_box(ax, 4, 3.5, 2, 1.2, 'Reserved', COLORS['aws'])
    draw_box(ax, 6.3, 3.5, 2, 1.2, 'Spot', COLORS['gcp'])

    # Approval
    draw_box(ax, 9, 4, 2, 2.5, 'Approval\nWorkflow\n\nAuto/Manual', COLORS['warning'])

    # Execution
    ax.text(12.5, 6.5, 'Execution', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 11.5, 4, 2.2, 2.5, 'Safe\nExecution\n\nRollback', COLORS['savings'])

    # Safety layer at bottom
    draw_box(ax, 2, 1, 10, 1.2, 'Safety Controls: Dry Run | Budget Limits | Approval Gates | Rollback', COLORS['dark'])

    # Arrows
    ax.annotate('', xy=(3.9, 5.3), xytext=(3.6, 5.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(8.9, 5.25), xytext=(8.4, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(11.4, 5.25), xytext=(11.1, 5.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_15_4_finops_automation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 15 figures...")
    create_multicloud_overview()
    print("  Created fig_15_1_multicloud_overview.png")
    create_cost_aggregation()
    print("  Created fig_15_2_cost_aggregation.png")
    create_workload_placement()
    print("  Created fig_15_3_workload_placement.png")
    create_finops_automation()
    print("  Created fig_15_4_finops_automation.png")
    print("Done!")


if __name__ == "__main__":
    main()
