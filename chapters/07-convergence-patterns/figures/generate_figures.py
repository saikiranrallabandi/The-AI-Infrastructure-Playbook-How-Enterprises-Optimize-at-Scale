#!/usr/bin/env python3
"""
Generate figures for Chapter 7: Cross-Industry Convergence Patterns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Arrow
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
    'ecommerce': '#FF6B35',
    'finance': '#2E86AB',
    'healthcare': '#28A745',
    'universal': '#6C5B7B',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'accent': '#FFC107'
}


def create_convergence_timeline():
    """Figure 7.1: Cross-Industry Pattern Convergence Timeline"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline years
    years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    ax.set_xlim(2009, 2025)
    ax.set_ylim(0, 5)

    # Draw timeline
    ax.axhline(y=2.5, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    # Patterns and their adoption by industry
    patterns = [
        ("Event Sourcing", 2010, "Trading", COLORS['finance']),
        ("Event Sourcing", 2015, "Banking", COLORS['finance']),
        ("Event Sourcing", 2020, "Healthcare", COLORS['healthcare']),
        ("Event Sourcing", 2022, "E-commerce", COLORS['ecommerce']),
        ("Microservices", 2012, "Netflix", COLORS['ecommerce']),
        ("Microservices", 2016, "Finance", COLORS['finance']),
        ("Microservices", 2018, "Healthcare", COLORS['healthcare']),
        ("Service Mesh", 2016, "Google", COLORS['universal']),
        ("Service Mesh", 2019, "Finance", COLORS['finance']),
        ("Service Mesh", 2022, "Healthcare", COLORS['healthcare']),
    ]

    # Group by pattern for vertical positioning
    pattern_y = {
        "Event Sourcing": 4,
        "Microservices": 2.5,
        "Service Mesh": 1
    }

    for pattern, year, industry, color in patterns:
        y = pattern_y[pattern]
        ax.scatter(year, y, s=200, c=color, zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(industry, (year, y), textcoords="offset points",
                   xytext=(0, 15), ha='center', fontsize=8, fontweight='bold')

    # Pattern labels on left
    for pattern, y in pattern_y.items():
        ax.text(2009.5, y, pattern, fontsize=11, fontweight='bold', va='center')

    # Year labels
    for year in years:
        ax.axvline(x=year, color='gray', linestyle='--', alpha=0.3)
        ax.text(year, 0.3, str(year), ha='center', fontsize=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['finance'], label='Finance'),
        mpatches.Patch(facecolor=COLORS['healthcare'], label='Healthcare'),
        mpatches.Patch(facecolor=COLORS['ecommerce'], label='E-commerce'),
        mpatches.Patch(facecolor=COLORS['universal'], label='Tech Giants'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title('Architecture Pattern Adoption Timeline Across Industries',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig_7_1_convergence_timeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_event_architecture():
    """Figure 7.2: Universal Event-Driven Architecture"""
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

    ax.text(7, 9.7, 'Universal Event-Driven Architecture', fontsize=14, fontweight='bold', ha='center')

    # Event Producers
    ax.text(0.3, 8.8, 'Event Producers', fontsize=10, fontweight='bold')
    draw_box(0.5, 7.8, 2, 0.8, 'E-commerce', COLORS['ecommerce'], 'Orders, Cart')
    draw_box(2.8, 7.8, 2, 0.8, 'Finance', COLORS['finance'], 'Trades, Payments')
    draw_box(5.1, 7.8, 2, 0.8, 'Healthcare', COLORS['healthcare'], 'Clinical Events')
    draw_box(7.4, 7.8, 2, 0.8, 'IoT', COLORS['universal'], 'Sensor Data')

    # Event Bus
    ax.text(0.3, 7.2, 'Event Bus', fontsize=10, fontweight='bold')
    bus = FancyBboxPatch((0.5, 6.2), 13, 0.8, boxstyle="round,pad=0.02",
                         facecolor=COLORS['primary'], edgecolor='white', linewidth=2)
    ax.add_patch(bus)
    ax.text(7, 6.6, 'Apache Kafka / Pulsar / Event Store', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Event Store
    ax.text(0.3, 5.6, 'Event Store', fontsize=10, fontweight='bold')
    draw_box(0.5, 4.6, 4, 0.8, 'Immutable Event Log', COLORS['dark'], 'Append-only')
    draw_box(5, 4.6, 4, 0.8, 'Schema Registry', COLORS['dark'], 'Avro/Protobuf')
    draw_box(9.5, 4.6, 4, 0.8, 'Replay Capability', COLORS['dark'], 'Point-in-time')

    # Stream Processing
    ax.text(0.3, 4.0, 'Stream Processing', fontsize=10, fontweight='bold')
    draw_box(0.5, 3.0, 3, 0.8, 'Flink/Spark', COLORS['secondary'], 'Aggregations')
    draw_box(4, 3.0, 3, 0.8, 'ksqlDB', COLORS['secondary'], 'SQL on streams')
    draw_box(7.5, 3.0, 3, 0.8, 'Custom', COLORS['secondary'], 'Domain logic')
    draw_box(11, 3.0, 2.5, 0.8, 'ML', COLORS['secondary'], 'Real-time')

    # Projections/Read Models
    ax.text(0.3, 2.4, 'Projections', fontsize=10, fontweight='bold')
    draw_box(0.5, 1.4, 2.5, 0.8, 'Search', COLORS['accent'], 'Elasticsearch')
    draw_box(3.3, 1.4, 2.5, 0.8, 'Analytics', COLORS['accent'], 'Data Warehouse')
    draw_box(6.1, 1.4, 2.5, 0.8, 'Cache', COLORS['accent'], 'Redis')
    draw_box(8.9, 1.4, 2.5, 0.8, 'API Views', COLORS['accent'], 'GraphQL')
    draw_box(11.7, 1.4, 1.8, 0.8, 'Reports', COLORS['accent'])

    # Consumers
    ax.text(0.3, 0.8, 'Consumers', fontsize=10, fontweight='bold')
    draw_box(0.5, 0.2, 3, 0.5, 'Web/Mobile Apps', COLORS['dark'])
    draw_box(4, 0.2, 3, 0.5, 'APIs', COLORS['dark'])
    draw_box(7.5, 0.2, 3, 0.5, 'Dashboards', COLORS['dark'])
    draw_box(11, 0.2, 2.5, 0.5, 'ML Models', COLORS['dark'])

    plt.tight_layout()
    plt.savefig('fig_7_2_event_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_universal_stack():
    """Figure 7.3: Universal Infrastructure Stack"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_layer(y, label, components, color):
        # Layer background
        layer = FancyBboxPatch((0.5, y), 11, 1.1, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(layer)

        # Layer label
        ax.text(1, y + 0.55, label, fontsize=10, fontweight='bold', color='white', va='center')

        # Components
        x_start = 3
        comp_width = 7.5 / len(components)
        for i, comp in enumerate(components):
            ax.text(x_start + i * comp_width + comp_width/2, y + 0.55, comp,
                   ha='center', va='center', fontsize=9, color='white')

    ax.text(6, 9.7, 'The Converged Enterprise Infrastructure Stack',
            fontsize=14, fontweight='bold', ha='center')

    # Stack layers (bottom to top)
    layers = [
        (0.5, "Infrastructure", ["Bare Metal", "VMs", "Cloud IaaS"], "#6C757D"),
        (1.8, "Orchestration", ["Kubernetes", "Nomad", "ECS"], COLORS['primary']),
        (3.1, "Service Mesh", ["Istio", "Linkerd", "Consul"], COLORS['secondary']),
        (4.4, "Messaging", ["Kafka", "Pulsar", "RabbitMQ"], COLORS['finance']),
        (5.7, "Data Stores", ["PostgreSQL", "MongoDB", "Redis"], COLORS['healthcare']),
        (7.0, "Observability", ["Prometheus", "Grafana", "Jaeger"], COLORS['ecommerce']),
        (8.3, "ML Platform", ["Kubeflow", "MLflow", "Seldon"], COLORS['universal']),
    ]

    for y, label, components, color in layers:
        draw_layer(y, label, components, color)

    # Industry badges on right
    ax.text(11.8, 5, 'Used by:', fontsize=9, fontweight='bold', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig('fig_7_3_universal_stack.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_prose_comparison():
    """Figure 7.4: Cross-Industry PROSE Comparison"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    categories = ['Performance', 'Resource\nOptimization', 'Operational\nExcellence',
                  'Scalability', 'Economic\nImpact']

    # Values (percentages converted to 0-100 scale)
    industries = {
        'E-Commerce': [30, 20, 20, 20, 10],
        'Finance': [25, 15, 30, 15, 15],
        'Healthcare': [20, 15, 35, 15, 15],
        'Universal': [25, 17, 28, 17, 13]
    }

    # Normalize to sum to 100 for display
    for key in industries:
        industries[key] = [v * 2 for v in industries[key]]  # Scale for visibility

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = [COLORS['ecommerce'], COLORS['finance'], COLORS['healthcare'], COLORS['universal']]

    for i, (industry, values) in enumerate(industries.items()):
        values = values + values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=industry, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 80)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(['10%', '20%', '30%'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.title('PROSE Framework Weights Across Industries', size=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig('fig_7_4_prose_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Generate all figures for Chapter 7."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Generating Chapter 7 figures...")

    print("  Creating Figure 7.1: Convergence Timeline...")
    create_convergence_timeline()

    print("  Creating Figure 7.2: Event-Driven Architecture...")
    create_event_architecture()

    print("  Creating Figure 7.3: Universal Stack...")
    create_universal_stack()

    print("  Creating Figure 7.4: PROSE Comparison...")
    create_prose_comparison()

    print("Done! All figures generated successfully.")


if __name__ == "__main__":
    main()
