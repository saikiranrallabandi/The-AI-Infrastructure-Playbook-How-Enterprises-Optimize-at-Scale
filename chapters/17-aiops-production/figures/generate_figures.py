#!/usr/bin/env python3
"""Generate figures for Chapter 17: AIOps Production"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'ml': '#6F42C1',
    'alert': '#DC3545',
    'remediate': '#28A745',
    'dark': '#343A40',
    'warning': '#FFC107',
    'data': '#17A2B8'
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


def create_aiops_architecture():
    """Figure 17.1: AIOps Production Architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'AIOps Production Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Data Layer
    ax.text(2, 6.8, 'Data Sources', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5.5, 1.5, 1, 'Metrics', COLORS['data'])
    draw_box(ax, 2, 5.5, 1.5, 1, 'Logs', COLORS['data'])
    draw_box(ax, 0.3, 4.2, 1.5, 1, 'Traces', COLORS['data'])
    draw_box(ax, 2, 4.2, 1.5, 1, 'Events', COLORS['data'])

    # Processing Layer
    ax.text(5.5, 6.8, 'ML Models', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 4, 5.5, 1.5, 1, 'Anomaly', COLORS['ml'])
    draw_box(ax, 5.7, 5.5, 1.5, 1, 'Forecast', COLORS['ml'])
    draw_box(ax, 4, 4.2, 1.5, 1, 'Classify', COLORS['ml'])
    draw_box(ax, 5.7, 4.2, 1.5, 1, 'Correlate', COLORS['ml'])

    # Decision Layer
    ax.text(9, 6.8, 'Decision Engine', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 7.8, 4.5, 2.5, 2, 'Orchestrator\n\nIncident\nManagement', COLORS['primary'])

    # Action Layer
    ax.text(12.5, 6.8, 'Actions', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 11, 5.5, 3, 1, 'Intelligent Alerts', COLORS['alert'])
    draw_box(ax, 11, 4.2, 3, 1, 'Auto Remediation', COLORS['remediate'])

    # Human in the loop
    draw_box(ax, 4, 1.5, 6, 1.5, 'Human-in-the-Loop: Approval Workflows | Escalation | Feedback', COLORS['dark'])

    # Arrows
    ax.annotate('', xy=(3.9, 5.5), xytext=(3.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(7.7, 5.5), xytext=(7.3, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(10.9, 5.5), xytext=(10.4, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_17_1_aiops_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_data_flow():
    """Figure 17.2: AIOps Data Flow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'AIOps Data Flow Pipeline',
            fontsize=14, fontweight='bold', ha='center')

    # Ingestion
    draw_box(ax, 0.5, 4.5, 2, 2.5, 'Ingestion\n\nAsync\nBuffered\nValidated', COLORS['data'])

    # Processing
    draw_box(ax, 3, 5.5, 2, 1.2, 'Transform', COLORS['secondary'])
    draw_box(ax, 3, 4, 2, 1.2, 'Enrich', COLORS['secondary'])

    # Feature Store
    draw_box(ax, 5.5, 4, 2.5, 2.8, 'Feature\nStore\n\nReal-time\nBatch\nHistorical', COLORS['dark'])

    # Inference
    draw_box(ax, 8.5, 5, 2.5, 2, 'Model\nInference', COLORS['ml'])

    # Outputs
    draw_box(ax, 11.5, 5.5, 2.2, 1.2, 'Predictions', COLORS['primary'])
    draw_box(ax, 11.5, 4, 2.2, 1.2, 'Actions', COLORS['remediate'])

    # Feedback loop
    draw_box(ax, 5, 1.5, 4, 1.2, 'Feedback Loop: Model Retraining', COLORS['warning'])

    # Arrows
    ax.annotate('', xy=(2.9, 5.5), xytext=(2.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(5.4, 5.3), xytext=(5.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(8.4, 5.5), xytext=(8.1, 5.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(11.4, 5.5), xytext=(11.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_17_2_data_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_model_lifecycle():
    """Figure 17.3: Model Lifecycle Management"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'ML Model Lifecycle for AIOps',
            fontsize=14, fontweight='bold', ha='center')

    # Stages
    stages = [
        ('Train', 'Data prep\nHypertuning', COLORS['data'], 0.5),
        ('Validate', 'Test metrics\nDrift check', COLORS['secondary'], 3.3),
        ('Stage', 'Shadow mode\nCanary', COLORS['warning'], 6.1),
        ('Production', 'Full traffic\nMonitor', COLORS['remediate'], 8.9),
        ('Retire', 'Archive\nRollback', COLORS['dark'], 11.7)
    ]

    for name, desc, color, x in stages:
        draw_box(ax, x, 4, 2.5, 2.5, f'{name}\n\n{desc}', color)

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1][3] - 0.1, 5.25),
                    xytext=(stages[i][3] + 2.6, 5.25),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Drift detection
    draw_box(ax, 2, 1, 10, 1.5, 'Continuous Monitoring: Data Drift | Model Degradation | Performance Alerts', COLORS['alert'])

    plt.tight_layout()
    plt.savefig('fig_17_3_model_lifecycle.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_remediation():
    """Figure 17.4: Auto-Remediation Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Safe Auto-Remediation Pipeline',
            fontsize=14, fontweight='bold', ha='center')

    # Detection
    draw_box(ax, 0.5, 4.5, 2.5, 2, 'Incident\nDetection', COLORS['alert'])

    # Risk Assessment
    draw_box(ax, 3.5, 4.5, 2.5, 2, 'Risk\nAssessment\n\nLow/Med/High', COLORS['warning'])

    # Decision
    ax.text(7.75, 6.5, 'Decision', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 6.5, 5, 2.5, 1.2, 'Auto Approve', COLORS['remediate'])
    draw_box(ax, 6.5, 3.5, 2.5, 1.2, 'Request Approval', COLORS['primary'])

    # Execution
    draw_box(ax, 9.5, 4, 2.5, 2.5, 'Execute\n\nCircuit Breaker\nRate Limit', COLORS['secondary'])

    # Rollback
    draw_box(ax, 12.5, 4.5, 1.2, 2, 'Verify\n\nRollback', COLORS['dark'])

    # Safety layer
    draw_box(ax, 2, 1, 10, 1.5, 'Safety Controls: Dry Run | Blast Radius | Approval Gates | Audit Log', COLORS['dark'])

    # Arrows
    ax.annotate('', xy=(3.4, 5.5), xytext=(3.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6.4, 5.5), xytext=(6.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(9.4, 5.25), xytext=(9.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(12.4, 5.5), xytext=(12.1, 5.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_17_4_remediation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 17 figures...")
    create_aiops_architecture()
    print("  Created fig_17_1_aiops_architecture.png")
    create_data_flow()
    print("  Created fig_17_2_data_flow.png")
    create_model_lifecycle()
    print("  Created fig_17_3_model_lifecycle.png")
    create_remediation()
    print("  Created fig_17_4_remediation.png")
    print("Done!")


if __name__ == "__main__":
    main()
