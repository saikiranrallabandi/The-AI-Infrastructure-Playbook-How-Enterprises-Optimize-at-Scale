#!/usr/bin/env python3
"""
Generate figures for Chapter 9: Building Your Optimization Platform
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'

# Color palette
COLORS = {
    'collection': '#4A90D9',
    'storage': '#28A745',
    'analysis': '#FFC107',
    'decision': '#E83E8C',
    'action': '#FF6B35',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545'
}


def create_platform_architecture():
    """Figure 9.1: Complete Optimization Platform Architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.15, label,
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)
        else:
            ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax.text(7, 9.7, 'Optimization Platform Architecture', fontsize=14, fontweight='bold', ha='center')

    # Layer 1: Data Collection
    ax.text(0.3, 8.8, 'Data Collection Layer', fontsize=10, fontweight='bold', color=COLORS['collection'])
    draw_box(0.5, 7.8, 2.5, 0.8, 'Metrics', COLORS['collection'], 'Prometheus')
    draw_box(3.3, 7.8, 2.5, 0.8, 'Logs', COLORS['collection'], 'Fluentd')
    draw_box(6.1, 7.8, 2.5, 0.8, 'Traces', COLORS['collection'], 'OpenTelemetry')
    draw_box(8.9, 7.8, 2.5, 0.8, 'Events', COLORS['collection'], 'Kafka')
    draw_box(11.7, 7.8, 1.8, 0.8, 'CMDB', COLORS['collection'])

    # Layer 2: Storage
    ax.text(0.3, 7.0, 'Storage Layer', fontsize=10, fontweight='bold', color=COLORS['storage'])
    draw_box(0.5, 6.0, 3, 0.8, 'Hot Storage', COLORS['storage'], 'InfluxDB (24h)')
    draw_box(4, 6.0, 3, 0.8, 'Warm Storage', COLORS['storage'], 'TimescaleDB (30d)')
    draw_box(7.5, 6.0, 3, 0.8, 'Cold Storage', COLORS['storage'], 'S3 Archive')
    draw_box(11, 6.0, 2.5, 0.8, 'Feature Store', COLORS['storage'], 'Redis')

    # Layer 3: Analysis
    ax.text(0.3, 5.2, 'Analysis Layer', fontsize=10, fontweight='bold', color=COLORS['analysis'])
    draw_box(0.5, 4.2, 3.2, 0.8, 'Stream Processing', COLORS['analysis'], 'Apache Flink')
    draw_box(4.2, 4.2, 3.2, 0.8, 'Batch Analysis', COLORS['analysis'], 'Spark')
    draw_box(7.9, 4.2, 2.8, 0.8, 'ML Models', COLORS['analysis'], 'Anomaly/Forecast')
    draw_box(11.2, 4.2, 2.3, 0.8, 'Alerts', COLORS['analysis'])

    # Layer 4: Decision
    ax.text(0.3, 3.4, 'Decision Layer', fontsize=10, fontweight='bold', color=COLORS['decision'])
    draw_box(0.5, 2.4, 3, 0.8, 'Policy Engine', COLORS['decision'], 'Rules + ML')
    draw_box(4, 2.4, 3, 0.8, 'Risk Assessor', COLORS['decision'], 'Safety checks')
    draw_box(7.5, 2.4, 3, 0.8, 'Optimizer', COLORS['decision'], 'Multi-objective')
    draw_box(11, 2.4, 2.5, 0.8, 'Approvals', COLORS['decision'])

    # Layer 5: Action
    ax.text(0.3, 1.6, 'Action Layer', fontsize=10, fontweight='bold', color=COLORS['action'])
    draw_box(0.5, 0.6, 2.5, 0.8, 'K8s Executor', COLORS['action'], 'Operators')
    draw_box(3.3, 0.6, 2.5, 0.8, 'Cloud APIs', COLORS['action'], 'AWS/GCP/Azure')
    draw_box(6.1, 0.6, 2.5, 0.8, 'Terraform', COLORS['action'], 'IaC')
    draw_box(8.9, 0.6, 2.5, 0.8, 'Rollback', COLORS['action'], 'Safety net')
    draw_box(11.7, 0.6, 1.8, 0.8, 'Audit', COLORS['action'])

    # Flow arrows
    for y in [7.7, 5.9, 4.1, 2.3]:
        ax.annotate('', xy=(7, y - 0.3), xytext=(7, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_9_1_platform_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_collection_pipeline():
    """Figure 9.2: Unified Data Collection Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.12, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(x + width/2, y + height/2 - 0.12, sublabel,
                    ha='center', va='center', fontsize=6, color='white', alpha=0.9)
        else:
            ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(7, 7.7, 'Unified Data Collection Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Sources (left)
    ax.text(1.5, 7, 'Data Sources', fontsize=10, fontweight='bold', ha='center')
    sources = [
        ('Prometheus', 6.2), ('CloudWatch', 5.4), ('Datadog', 4.6),
        ('App Logs', 3.8), ('K8s Events', 3.0), ('Traces', 2.2)
    ]
    for name, y in sources:
        draw_box(0.3, y, 2.4, 0.6, name, COLORS['collection'])

    # Collectors
    ax.text(4.5, 7, 'Collectors', fontsize=10, fontweight='bold', ha='center')
    draw_box(3.5, 4.5, 2, 2, 'Metrics\nCollector', COLORS['primary'])
    draw_box(3.5, 2.2, 2, 2, 'Log\nCollector', COLORS['primary'])

    # Normalizer
    ax.text(7.5, 7, 'Processing', fontsize=10, fontweight='bold', ha='center')
    draw_box(6.5, 3.5, 2, 2.5, 'Normalizer\n\nSchema\nValidation', COLORS['secondary'])

    # Buffer
    draw_box(9.5, 3.5, 2, 2.5, 'Buffer\n\nBack-\npressure', COLORS['success'])

    # Output
    ax.text(12.5, 7, 'Output', fontsize=10, fontweight='bold', ha='center')
    draw_box(11.8, 5, 1.9, 1.2, 'Storage', COLORS['storage'])
    draw_box(11.8, 3.5, 1.9, 1.2, 'Stream', COLORS['analysis'])
    draw_box(11.8, 2, 1.9, 1.2, 'Alerts', COLORS['warning'])

    # Arrows
    for y in [6.5, 5.7, 4.9]:
        ax.annotate('', xy=(3.4, 5.5), xytext=(2.8, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    for y in [4.1, 3.3, 2.5]:
        ax.annotate('', xy=(3.4, 3.2), xytext=(2.8, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.annotate('', xy=(6.4, 4.75), xytext=(5.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(6.4, 4.75), xytext=(5.6, 3.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(9.4, 4.75), xytext=(8.6, 4.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    for y in [5.6, 4.1, 2.6]:
        ax.annotate('', xy=(11.7, y), xytext=(11.6, 4.75),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()
    plt.savefig('fig_9_2_collection_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_analysis_engine():
    """Figure 9.3: Multi-Layer Analysis Engine"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.12, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(x + width/2, y + height/2 - 0.12, sublabel,
                    ha='center', va='center', fontsize=6, color='white', alpha=0.9)
        else:
            ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(7, 7.7, 'Multi-Layer Analysis Engine', fontsize=14, fontweight='bold', ha='center')

    # Stream Processing (top)
    ax.text(7, 7.0, 'Real-Time Stream Processing', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 5.8, 2.5, 1, 'Anomaly\nDetection', COLORS['warning'])
    draw_box(3.3, 5.8, 2.5, 1, 'Pattern\nMatching', COLORS['warning'])
    draw_box(6.1, 5.8, 2.5, 1, 'Threshold\nAlerts', COLORS['warning'])
    draw_box(8.9, 5.8, 2.5, 1, 'Trend\nDetection', COLORS['warning'])
    draw_box(11.7, 5.8, 1.8, 1, 'Correlation', COLORS['warning'])

    # Batch Processing (middle)
    ax.text(7, 4.8, 'Batch Analysis', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 3.6, 3, 1, 'Capacity\nForecasting', COLORS['primary'])
    draw_box(4, 3.6, 3, 1, 'Cost\nAnalysis', COLORS['primary'])
    draw_box(7.5, 3.6, 3, 1, 'Performance\nBaselining', COLORS['primary'])
    draw_box(11, 3.6, 2.5, 1, 'Report\nGeneration', COLORS['primary'])

    # ML Models (bottom)
    ax.text(7, 2.6, 'ML Model Layer', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 1.4, 2.5, 1, 'Isolation\nForest', COLORS['secondary'])
    draw_box(3.3, 1.4, 2.5, 1, 'LSTM\nPredictor', COLORS['secondary'])
    draw_box(6.1, 1.4, 2.5, 1, 'Autoencoder', COLORS['secondary'])
    draw_box(8.9, 1.4, 2.5, 1, 'Ensemble', COLORS['secondary'])
    draw_box(11.7, 1.4, 1.8, 1, 'Online\nLearning', COLORS['secondary'])

    # Feature Store connection
    draw_box(5.5, 0.2, 3, 0.8, 'Feature Store', COLORS['success'])

    plt.tight_layout()
    plt.savefig('fig_9_3_analysis_engine.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_action_pipeline():
    """Figure 9.4: Safe Action Execution Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.12, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(x + width/2, y + height/2 - 0.12, sublabel,
                    ha='center', va='center', fontsize=6, color='white', alpha=0.9)
        else:
            ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(7, 7.7, 'Safe Action Execution Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Main flow
    draw_box(0.5, 5.5, 2, 1.5, 'Decision\nReceived', COLORS['decision'])

    draw_box(3, 5.5, 2, 1.5, 'Pre-flight\nSafety\nChecks', COLORS['danger'])

    draw_box(5.5, 5.5, 2, 1.5, 'Create\nRollback\nPoint', COLORS['warning'])

    draw_box(8, 5.5, 2, 1.5, 'Execute\nAction', COLORS['action'])

    draw_box(10.5, 5.5, 2, 1.5, 'Monitor\nExecution', COLORS['primary'])

    # Success path
    draw_box(10.5, 3.5, 2, 1, 'Success\nAudit Log', COLORS['success'])

    # Failure path
    draw_box(8, 3.5, 2, 1, 'Rollback', COLORS['danger'])

    # Safety checks detail
    ax.text(4, 4.8, 'Safety Checks:', fontsize=8, fontweight='bold')
    checks = ['Lockout status', 'Rate limits', 'Dependencies', 'Business hours']
    for i, check in enumerate(checks):
        ax.text(4, 4.4 - i*0.35, f'• {check}', fontsize=7)

    # Arrows
    ax.annotate('', xy=(2.9, 6.25), xytext=(2.6, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(5.4, 6.25), xytext=(5.1, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(7.9, 6.25), xytext=(7.6, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(10.4, 6.25), xytext=(10.1, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Success arrow
    ax.annotate('', xy=(11.5, 4.6), xytext=(11.5, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    ax.text(12, 5, 'Success', fontsize=7, color=COLORS['success'])

    # Failure arrow
    ax.annotate('', xy=(9, 4.6), xytext=(10.4, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2))
    ax.text(9.5, 5, 'Failure', fontsize=7, color=COLORS['danger'])

    plt.tight_layout()
    plt.savefig('fig_9_4_action_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Generate all figures for Chapter 9."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Generating Chapter 9 figures...")

    print("  Creating Figure 9.1: Platform Architecture...")
    create_platform_architecture()

    print("  Creating Figure 9.2: Collection Pipeline...")
    create_collection_pipeline()

    print("  Creating Figure 9.3: Analysis Engine...")
    create_analysis_engine()

    print("  Creating Figure 9.4: Action Pipeline...")
    create_action_pipeline()

    print("Done! All figures generated successfully.")


if __name__ == "__main__":
    main()
