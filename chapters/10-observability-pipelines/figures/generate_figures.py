#!/usr/bin/env python3
"""
Generate figures for Chapter 10: Observability and Data Pipelines
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'metrics': '#4A90D9',
    'logs': '#28A745',
    'traces': '#E83E8C',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'success': '#28A745'
}


def draw_box(ax, x, y, width, height, label, color, sublabel=None):
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


def create_metrics_pipeline():
    """Figure 10.1: Enterprise Metrics Pipeline Architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Enterprise Metrics Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Sources
    ax.text(1.5, 7, 'Sources', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5.8, 2.4, 0.9, 'Prometheus', COLORS['metrics'], 'Pull-based')
    draw_box(ax, 0.3, 4.6, 2.4, 0.9, 'CloudWatch', COLORS['metrics'], 'AWS')
    draw_box(ax, 0.3, 3.4, 2.4, 0.9, 'Datadog', COLORS['metrics'], 'SaaS')
    draw_box(ax, 0.3, 2.2, 2.4, 0.9, 'StatsD', COLORS['metrics'], 'Push')

    # Collection
    ax.text(4.5, 7, 'Collection', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.5, 3.5, 2, 2.5, 'Unified\nCollector', COLORS['primary'])

    # Processing
    ax.text(7.5, 7, 'Processing', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 6.5, 5.2, 2, 1.3, 'Validation', COLORS['secondary'])
    draw_box(ax, 6.5, 3.5, 2, 1.3, 'Normalization', COLORS['secondary'])
    draw_box(ax, 6.5, 1.8, 2, 1.3, 'Aggregation', COLORS['secondary'])

    # Storage
    ax.text(10.5, 7, 'Storage', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 9.5, 5.5, 2, 1, 'Hot (24h)', COLORS['success'], 'InfluxDB')
    draw_box(ax, 9.5, 4.2, 2, 1, 'Warm (30d)', COLORS['warning'], 'TimescaleDB')
    draw_box(ax, 9.5, 2.9, 2, 1, 'Cold', COLORS['dark'], 'S3')

    # Output
    ax.text(13, 7, 'Output', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 12, 5.5, 1.7, 0.9, 'Dashboards', COLORS['primary'])
    draw_box(ax, 12, 4.3, 1.7, 0.9, 'Alerts', COLORS['danger'])
    draw_box(ax, 12, 3.1, 1.7, 0.9, 'ML Pipeline', COLORS['secondary'])

    plt.tight_layout()
    plt.savefig('fig_10_1_metrics_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_three_pillars():
    """Figure 10.2: Integrated Three Pillars of Observability"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6, 7.7, 'Three Pillars of Observability', fontsize=14, fontweight='bold', ha='center')

    # Three pillars
    draw_box(ax, 0.5, 4, 3, 3, 'METRICS\n\nTime-series\nNumerical data\nAggregations', COLORS['metrics'])
    draw_box(ax, 4.5, 4, 3, 3, 'LOGS\n\nEvent records\nContext\nSearchable', COLORS['logs'])
    draw_box(ax, 8.5, 4, 3, 3, 'TRACES\n\nRequest flow\nLatency\nDependencies', COLORS['traces'])

    # Correlation layer
    draw_box(ax, 2, 1.5, 8, 1.5, 'Correlation Layer - Unified Context via Trace ID, Timestamps, Labels', COLORS['dark'])

    # Arrows to correlation
    for x in [2, 6, 10]:
        ax.annotate('', xy=(6, 3.2), xytext=(x, 3.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_10_2_three_pillars.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_pipeline_architecture():
    """Figure 10.3: Complete Observability Data Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Observability Data Pipeline Architecture', fontsize=14, fontweight='bold', ha='center')

    # Ingestion
    draw_box(ax, 0.5, 5.5, 2.5, 1.5, 'Ingestion\nLayer', COLORS['primary'])
    draw_box(ax, 0.5, 3.5, 2.5, 1.5, 'Kafka\nBuffer', COLORS['secondary'])

    # Stream Processing
    draw_box(ax, 3.5, 4, 2.5, 3, 'Stream\nProcessing\n\nFlink/Spark', COLORS['warning'])

    # Storage Tiers
    draw_box(ax, 6.5, 5.5, 2.5, 1.5, 'Hot Storage', COLORS['success'])
    draw_box(ax, 6.5, 3.5, 2.5, 1.5, 'Warm Storage', COLORS['warning'])
    draw_box(ax, 6.5, 1.5, 2.5, 1.5, 'Cold Storage', COLORS['dark'])

    # Query Layer
    draw_box(ax, 9.5, 3.5, 2, 3, 'Query\nLayer\n\nUnified API', COLORS['primary'])

    # Consumers
    draw_box(ax, 12, 5.5, 1.7, 1, 'Dashboards', COLORS['metrics'])
    draw_box(ax, 12, 4.2, 1.7, 1, 'Alerts', COLORS['danger'])
    draw_box(ax, 12, 2.9, 1.7, 1, 'ML/AI', COLORS['secondary'])

    # Arrows
    ax.annotate('', xy=(3.4, 5.5), xytext=(3.1, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(6.4, 5.5), xytext=(6.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(9.4, 5), xytext=(9.1, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(11.9, 5), xytext=(11.6, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_10_3_pipeline_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_anomaly_detection():
    """Figure 10.4: ML-Based Anomaly Detection Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'ML-Based Anomaly Detection Pipeline', fontsize=14, fontweight='bold', ha='center')

    # Input
    draw_box(ax, 0.5, 4, 2, 2, 'Time Series\nData', COLORS['primary'])

    # Detectors
    ax.text(5, 7, 'Detector Ensemble', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.2, 5.5, 2, 1.2, 'Statistical', COLORS['secondary'], 'Z-Score, MAD')
    draw_box(ax, 5.5, 5.5, 2, 1.2, 'Isolation Forest', COLORS['secondary'], 'Tree-based')
    draw_box(ax, 3.2, 3.8, 2, 1.2, 'LSTM', COLORS['secondary'], 'Temporal')
    draw_box(ax, 5.5, 3.8, 2, 1.2, 'Autoencoder', COLORS['secondary'], 'Reconstruction')

    # Ensemble
    draw_box(ax, 8, 4.2, 2, 2, 'Ensemble\nScoring', COLORS['warning'])

    # Output
    draw_box(ax, 10.5, 5.5, 3, 1.2, 'Anomaly Alerts', COLORS['danger'])
    draw_box(ax, 10.5, 3.8, 3, 1.2, 'Explanations', COLORS['success'])

    # Arrows
    ax.annotate('', xy=(3.1, 5), xytext=(2.6, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7.9, 5.2), xytext=(7.6, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(10.4, 5.2), xytext=(10.1, 5.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_10_4_anomaly_detection.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Generate all figures for Chapter 10."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Generating Chapter 10 figures...")
    print("  Creating Figure 10.1: Metrics Pipeline...")
    create_metrics_pipeline()
    print("  Creating Figure 10.2: Three Pillars...")
    create_three_pillars()
    print("  Creating Figure 10.3: Pipeline Architecture...")
    create_pipeline_architecture()
    print("  Creating Figure 10.4: Anomaly Detection...")
    create_anomaly_detection()
    print("Done!")


if __name__ == "__main__":
    main()
