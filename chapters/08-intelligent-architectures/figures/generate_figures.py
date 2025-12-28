#!/usr/bin/env python3
"""
Generate figures for Chapter 8: Intelligent Infrastructure Architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
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
    'level0': '#DC3545',    # Red - Manual
    'level1': '#FD7E14',    # Orange - Scripted
    'level2': '#FFC107',    # Yellow - Reactive
    'level3': '#28A745',    # Green - Predictive
    'level4': '#17A2B8',    # Cyan - Adaptive
    'level5': '#6F42C1',    # Purple - Autonomous
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'accent': '#FF6B35'
}


def create_intelligence_levels():
    """Figure 8.1: Infrastructure Intelligence Maturity Model"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'Infrastructure Intelligence Maturity Model',
            fontsize=14, fontweight='bold', ha='center')

    levels = [
        (0, "Manual", "Human-driven", COLORS['level0'], "SSH, manual fixes"),
        (1, "Scripted", "Rule-based", COLORS['level1'], "Cron jobs, scripts"),
        (2, "Reactive", "Event-driven", COLORS['level2'], "Auto-restart"),
        (3, "Predictive", "ML forecasting", COLORS['level3'], "Failure prediction"),
        (4, "Adaptive", "Self-optimizing", COLORS['level4'], "Config tuning"),
        (5, "Autonomous", "Self-healing", COLORS['level5'], "Full autonomy"),
    ]

    # Draw pyramid/staircase
    for i, (level, name, desc, color, example) in enumerate(levels):
        # Calculate position - pyramid shape
        x_start = 1 + i * 0.5
        width = 12 - i * 1
        y = 1 + i * 1.2
        height = 1

        box = FancyBboxPatch(
            (x_start, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(box)

        # Level number
        ax.text(x_start + 0.3, y + 0.5, f"L{level}",
                fontsize=11, fontweight='bold', color='white', va='center')

        # Name and description
        ax.text(x_start + width/2, y + 0.6, name,
                fontsize=11, fontweight='bold', color='white', ha='center', va='center')
        ax.text(x_start + width/2, y + 0.3, desc,
                fontsize=9, color='white', ha='center', va='center', alpha=0.9)

        # Example on right
        ax.text(x_start + width - 0.3, y + 0.5, example,
                fontsize=8, color='white', ha='right', va='center', style='italic')

    # Arrow showing progression
    ax.annotate('', xy=(13.5, 7.5), xytext=(13.5, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(13.7, 4.5, 'Intelligence\nMaturity', fontsize=9, va='center', rotation=90)

    # Most enterprises marker
    ax.annotate('Most enterprises today', xy=(3, 3.8), xytext=(0.5, 5.5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_8_1_intelligence_levels.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_predictive_architecture():
    """Figure 8.2: Predictive Intelligence System Architecture"""
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
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    ax.text(7, 9.7, 'Predictive Intelligence System Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Data Sources (top)
    ax.text(7, 9.0, 'Data Sources', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 8.0, 2.5, 0.8, 'Metrics', COLORS['primary'], 'Prometheus')
    draw_box(3.3, 8.0, 2.5, 0.8, 'Logs', COLORS['primary'], 'ELK Stack')
    draw_box(6.1, 8.0, 2.5, 0.8, 'Traces', COLORS['primary'], 'Jaeger')
    draw_box(8.9, 8.0, 2.5, 0.8, 'Events', COLORS['primary'], 'Kafka')
    draw_box(11.7, 8.0, 1.8, 0.8, 'CMDB', COLORS['primary'])

    # Feature Engineering
    ax.text(7, 7.3, 'Feature Engineering', fontsize=10, fontweight='bold', ha='center')
    draw_box(1, 6.3, 4, 0.8, 'Feature Store', COLORS['secondary'], 'Time-series features')
    draw_box(5.5, 6.3, 4, 0.8, 'Anomaly Features', COLORS['secondary'], 'Statistical + ML')
    draw_box(10, 6.3, 3.5, 0.8, 'Pattern Features', COLORS['secondary'], 'Sequences')

    # ML Models
    ax.text(7, 5.6, 'ML Models', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 4.4, 2.5, 1, 'Anomaly\nDetection', COLORS['level3'])
    draw_box(3.3, 4.4, 2.5, 1, 'Failure\nPrediction', COLORS['level3'])
    draw_box(6.1, 4.4, 2.5, 1, 'Capacity\nForecasting', COLORS['level3'])
    draw_box(8.9, 4.4, 2.5, 1, 'Root Cause\nAnalysis', COLORS['level3'])
    draw_box(11.7, 4.4, 1.8, 1, 'Cost\nOptimizer', COLORS['level3'])

    # Decision Engine
    ax.text(7, 3.7, 'Decision Engine', fontsize=10, fontweight='bold', ha='center')
    draw_box(2, 2.6, 5, 0.9, 'Risk Assessment', COLORS['level4'], 'Confidence scoring')
    draw_box(7.5, 2.6, 5, 0.9, 'Action Selection', COLORS['level4'], 'Multi-objective optimization')

    # Actions
    ax.text(7, 1.9, 'Automated Actions', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 0.8, 2.5, 0.8, 'Alerts', COLORS['accent'])
    draw_box(3.3, 0.8, 2.5, 0.8, 'Auto-Scale', COLORS['accent'])
    draw_box(6.1, 0.8, 2.5, 0.8, 'Remediation', COLORS['accent'])
    draw_box(8.9, 0.8, 2.5, 0.8, 'Optimization', COLORS['accent'])
    draw_box(11.7, 0.8, 1.8, 0.8, 'Reports', COLORS['accent'])

    # Draw flow arrows
    for i in range(5):
        x = 1.75 + i * 2.8
        ax.annotate('', xy=(x, 7.9), xytext=(x, 7.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.annotate('', xy=(7, 6.2), xytext=(7, 5.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7, 4.3), xytext=(7, 3.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7, 2.5), xytext=(7, 2.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_8_2_predictive_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_self_healing():
    """Figure 8.3: Self-Healing Infrastructure Architecture"""
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

    ax.text(7, 9.7, 'Self-Healing Infrastructure Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # MAPE-K Loop (center)
    ax.text(7, 9.0, 'MAPE-K Control Loop', fontsize=11, fontweight='bold', ha='center')

    # Draw circular flow
    loop_center = (7, 5.5)
    loop_radius = 2.5

    # MAPE components
    mape_components = [
        ('Monitor', 90, COLORS['primary']),
        ('Analyze', 0, COLORS['secondary']),
        ('Plan', 270, COLORS['level4']),
        ('Execute', 180, COLORS['accent']),
    ]

    for name, angle, color in mape_components:
        rad = np.radians(angle)
        x = loop_center[0] + loop_radius * np.cos(rad)
        y = loop_center[1] + loop_radius * np.sin(rad)
        draw_box(x - 0.9, y - 0.4, 1.8, 0.8, name, color)

    # Knowledge base in center
    draw_box(loop_center[0] - 1.2, loop_center[1] - 0.5, 2.4, 1, 'Knowledge\nBase', COLORS['level5'])

    # Draw circular arrows between components
    for i, (_, angle, _) in enumerate(mape_components):
        next_angle = mape_components[(i + 1) % 4][1]
        start_rad = np.radians(angle - 20)
        end_rad = np.radians(next_angle + 20)

        start_x = loop_center[0] + (loop_radius - 0.3) * np.cos(start_rad)
        start_y = loop_center[1] + (loop_radius - 0.3) * np.sin(start_rad)
        end_x = loop_center[0] + (loop_radius - 0.3) * np.cos(end_rad)
        end_y = loop_center[1] + (loop_radius - 0.3) * np.sin(end_rad)

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                    connectionstyle='arc3,rad=0.3'))

    # Runbook Store (left)
    ax.text(1.5, 8.5, 'Runbooks', fontsize=10, fontweight='bold', ha='center')
    draw_box(0.5, 7.5, 2, 0.7, 'Disk Full', COLORS['dark'])
    draw_box(0.5, 6.6, 2, 0.7, 'High CPU', COLORS['dark'])
    draw_box(0.5, 5.7, 2, 0.7, 'Memory Leak', COLORS['dark'])
    draw_box(0.5, 4.8, 2, 0.7, 'Network', COLORS['dark'])

    # Chaos Engineering (right)
    ax.text(12.5, 8.5, 'Chaos Tests', fontsize=10, fontweight='bold', ha='center')
    draw_box(11.5, 7.5, 2, 0.7, 'Pod Kill', COLORS['level0'])
    draw_box(11.5, 6.6, 2, 0.7, 'Latency', COLORS['level0'])
    draw_box(11.5, 5.7, 2, 0.7, 'CPU Stress', COLORS['level0'])
    draw_box(11.5, 4.8, 2, 0.7, 'Network', COLORS['level0'])

    # Safety layer (bottom)
    ax.text(7, 2.5, 'Safety Layer', fontsize=10, fontweight='bold', ha='center')
    draw_box(1, 1.3, 3.5, 0.9, 'Approval Engine', COLORS['level2'], 'Human-in-loop')
    draw_box(5.25, 1.3, 3.5, 0.9, 'Guardrails', COLORS['level2'], 'Constraints')
    draw_box(9.5, 1.3, 3.5, 0.9, 'Audit Trail', COLORS['level2'], 'Immutable log')

    # Arrows from runbooks to MAPE
    ax.annotate('', xy=(4.2, 5.5), xytext=(2.7, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Arrows from chaos to MAPE
    ax.annotate('', xy=(9.8, 5.5), xytext=(11.3, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Arrow to safety
    ax.annotate('', xy=(7, 2.5), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_8_3_self_healing.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_architecture_patterns():
    """Figure 8.4: Intelligence Integration Patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    def draw_box(ax, x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=6, color='white', alpha=0.9)
        else:
            ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Pattern 1: Sidecar
    ax = axes[0]
    ax.set_title('Sidecar Pattern', fontsize=11, fontweight='bold', pad=10)

    # Main service
    draw_box(ax, 2, 4, 3, 3, 'Application\nService', COLORS['primary'])

    # Sidecar
    draw_box(ax, 5.5, 4, 2.5, 3, 'Intelligent\nSidecar', COLORS['secondary'])

    # Sidecar components
    draw_box(ax, 5.7, 5.8, 2.1, 0.8, 'Anomaly', COLORS['level3'])
    draw_box(ax, 5.7, 4.8, 2.1, 0.8, 'Optimizer', COLORS['level4'])
    draw_box(ax, 5.7, 3.8, 2.1, 0.8, 'Rate Limit', COLORS['accent'])

    # Traffic flow
    ax.annotate('', xy=(2, 5.5), xytext=(0.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1.25, 6, 'Traffic', fontsize=7, ha='center')

    ax.annotate('', xy=(5, 5.5), xytext=(5.3, 5.5),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

    # Pattern 2: Central Intelligence
    ax = axes[1]
    ax.set_title('Central Intelligence', fontsize=11, fontweight='bold', pad=10)

    # Central brain
    draw_box(ax, 3, 6, 4, 2, 'Intelligence\nHub', COLORS['level5'])

    # Services
    draw_box(ax, 0.5, 2, 2.5, 1.5, 'Service A', COLORS['primary'])
    draw_box(ax, 3.75, 2, 2.5, 1.5, 'Service B', COLORS['primary'])
    draw_box(ax, 7, 2, 2.5, 1.5, 'Service C', COLORS['primary'])

    # Arrows
    for x in [1.75, 5, 8.25]:
        ax.annotate('', xy=(5, 5.8), xytext=(x, 3.7),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1))

    # Pattern 3: Hierarchical
    ax = axes[2]
    ax.set_title('Hierarchical Pattern', fontsize=11, fontweight='bold', pad=10)

    # Global controller
    draw_box(ax, 3, 7.5, 4, 1.2, 'Global\nController', COLORS['level5'])

    # Regional controllers
    draw_box(ax, 1, 5, 3, 1.2, 'Region A', COLORS['level4'])
    draw_box(ax, 6, 5, 3, 1.2, 'Region B', COLORS['level4'])

    # Local agents
    draw_box(ax, 0.5, 2.5, 1.5, 1, 'Agent', COLORS['level3'])
    draw_box(ax, 2.5, 2.5, 1.5, 1, 'Agent', COLORS['level3'])
    draw_box(ax, 5.5, 2.5, 1.5, 1, 'Agent', COLORS['level3'])
    draw_box(ax, 7.5, 2.5, 1.5, 1, 'Agent', COLORS['level3'])

    # Arrows
    ax.annotate('', xy=(2.5, 6.3), xytext=(4, 7.3),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.annotate('', xy=(7.5, 6.3), xytext=(6, 7.3),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))

    for x in [1.25, 3.25]:
        ax.annotate('', xy=(x, 3.7), xytext=(2.5, 4.8),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    for x in [6.25, 8.25]:
        ax.annotate('', xy=(x, 3.7), xytext=(7.5, 4.8),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1))

    plt.suptitle('Intelligence Integration Patterns', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_8_4_architecture_patterns.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Generate all figures for Chapter 8."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Generating Chapter 8 figures...")

    print("  Creating Figure 8.1: Intelligence Levels...")
    create_intelligence_levels()

    print("  Creating Figure 8.2: Predictive Architecture...")
    create_predictive_architecture()

    print("  Creating Figure 8.3: Self-Healing Architecture...")
    create_self_healing()

    print("  Creating Figure 8.4: Architecture Patterns...")
    create_architecture_patterns()

    print("Done! All figures generated successfully.")


if __name__ == "__main__":
    main()
