#!/usr/bin/env python3
"""
Generate figures for Chapter 5: Financial Services Infrastructure Optimization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Color palette - financial services themed
COLORS = {
    'primary': '#1e3a5f',      # Dark blue (trust)
    'secondary': '#2e7d32',    # Green (money)
    'accent': '#c62828',       # Red (alerts)
    'warning': '#f57c00',      # Orange
    'info': '#0277bd',         # Light blue
    'success': '#388e3c',      # Success green
    'light': '#eceff1',        # Light gray
    'dark': '#263238',         # Dark gray
    'gold': '#ffc107'          # Gold
}

def save_figure(fig, name: str):
    """Save figure with consistent settings."""
    output_path = Path(__file__).parent / f"{name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_requirements_radar():
    """Figure 5.1: Financial Services Infrastructure Requirements by Domain."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Categories
    categories = ['Latency', 'Availability', 'Compliance', 'Security',
                  'Auditability', 'Scalability', 'Data Integrity']
    N = len(categories)

    # Data for different domains
    domains = {
        'Trading': [95, 85, 75, 90, 70, 60, 95],
        'Retail Banking': [60, 95, 90, 95, 95, 85, 99],
        'Payments': [80, 99, 85, 95, 90, 90, 99],
        'Risk Management': [70, 85, 80, 85, 85, 75, 95]
    }

    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Colors for domains
    domain_colors = [COLORS['primary'], COLORS['secondary'], COLORS['info'], COLORS['warning']]

    for (domain, values), color in zip(domains.items(), domain_colors):
        values = values + values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=domain, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Financial Services Infrastructure Requirements by Domain',
              size=14, fontweight='bold', y=1.08)

    save_figure(fig, 'fig_5_1_requirements_radar')


def create_compliance_architecture():
    """Figure 5.2: Financial Services Compliance Architecture."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        text_color = 'white' if color not in [COLORS['light'], COLORS['gold']] else COLORS['dark']
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9, color=text_color)
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=7, color=text_color, alpha=0.9)

    # Title
    ax.text(7, 9.5, 'Financial Services Compliance Architecture',
            ha='center', fontsize=14, fontweight='bold')

    # Layer 1: External Regulators
    ax.text(0.5, 8.7, 'Regulatory Bodies', fontsize=11, fontweight='bold')
    draw_box(2, 8.3, 2.3, 0.7, 'SEC/FINRA', COLORS['accent'])
    draw_box(4.5, 8.3, 2.3, 0.7, 'OCC/Fed', COLORS['accent'])
    draw_box(7, 8.3, 2.3, 0.7, 'PCI Council', COLORS['accent'])
    draw_box(9.5, 8.3, 2.3, 0.7, 'GDPR Auth', COLORS['accent'])

    # Layer 2: Compliance Gateway
    ax.text(0.5, 7.4, 'Compliance Layer', fontsize=11, fontweight='bold')
    draw_box(2, 6.8, 3.5, 0.9, 'Audit Log Aggregator', COLORS['primary'], 'Immutable storage')
    draw_box(6, 6.8, 3.5, 0.9, 'Policy Engine', COLORS['primary'], 'Real-time enforcement')
    draw_box(10, 6.8, 3, 0.9, 'Reporting System', COLORS['primary'], 'Auto reports')

    # Layer 3: Security Controls
    ax.text(0.5, 5.9, 'Security Controls', fontsize=11, fontweight='bold')
    draw_box(1.5, 5.3, 2.5, 0.9, 'IAM/PAM', COLORS['info'], 'Identity mgmt')
    draw_box(4.3, 5.3, 2.5, 0.9, 'Encryption', COLORS['info'], 'At rest + transit')
    draw_box(7.1, 5.3, 2.5, 0.9, 'DLP', COLORS['info'], 'Data protection')
    draw_box(9.9, 5.3, 2.5, 0.9, 'SIEM', COLORS['info'], 'Threat detection')

    # Layer 4: Application Layer
    ax.text(0.5, 4.4, 'Applications', fontsize=11, fontweight='bold')
    draw_box(1.5, 3.8, 2.2, 0.9, 'Trading', COLORS['secondary'])
    draw_box(4, 3.8, 2.2, 0.9, 'Banking', COLORS['secondary'])
    draw_box(6.5, 3.8, 2.2, 0.9, 'Payments', COLORS['secondary'])
    draw_box(9, 3.8, 2.2, 0.9, 'Risk', COLORS['secondary'])
    draw_box(11.5, 3.8, 2, 0.9, 'Analytics', COLORS['secondary'])

    # Layer 5: Data Layer
    ax.text(0.5, 2.9, 'Data Layer', fontsize=11, fontweight='bold')
    draw_box(2, 2.2, 3, 0.9, 'Transaction DB', COLORS['dark'], 'ACID compliant')
    draw_box(5.5, 2.2, 3, 0.9, 'Audit Store', COLORS['dark'], 'Immutable')
    draw_box(9, 2.2, 3, 0.9, 'Data Warehouse', COLORS['dark'], 'Encrypted')

    # Layer 6: Infrastructure
    ax.text(0.5, 1.4, 'Infrastructure', fontsize=11, fontweight='bold')
    draw_box(2, 0.6, 2.5, 0.9, 'On-Prem DC', COLORS['warning'], 'Regulated data')
    draw_box(5, 0.6, 2.5, 0.9, 'Private Cloud', COLORS['warning'], 'Hybrid')
    draw_box(8, 0.6, 2.5, 0.9, 'Public Cloud', COLORS['warning'], 'Non-sensitive')
    draw_box(11, 0.6, 2.5, 0.9, 'DR Site', COLORS['warning'], 'Failover')

    # Compliance badges - moved to the right
    badges = [('SOX', 13.8, 8.5), ('PCI-DSS', 13.8, 7.6), ('GDPR', 13.8, 6.7)]
    for badge, x, y in badges:
        circle = plt.Circle((x, y), 0.4, color=COLORS['gold'], ec=COLORS['dark'])
        ax.add_patch(circle)
        ax.text(x, y, badge, ha='center', va='center', fontsize=7, fontweight='bold')

    save_figure(fig, 'fig_5_2_compliance_architecture')


def create_latency_breakdown():
    """Figure 5.3: Trading System Latency Components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Waterfall chart of latency components
    ax = axes[0]
    components = ['Network\n(co-located)', 'Market Data\nParsing', 'Order\nValidation',
                  'Risk Check', 'Order\nRouting', 'Exchange\nAck']
    typical = [75, 30, 60, 100, 40, 50]
    optimized = [8, 3, 12, 25, 8, 15]

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax.bar(x - width/2, typical, width, label='Typical', color=COLORS['warning'], alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized, width, label='Optimized', color=COLORS['success'], alpha=0.8)

    ax.set_ylabel('Latency (microseconds)')
    ax.set_title('Trading System Latency by Component', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=9)
    ax.legend()

    # Add total labels
    ax.text(5.5, max(typical) * 0.9, f'Total: {sum(typical)}us', fontsize=10,
            color=COLORS['warning'], fontweight='bold')
    ax.text(5.5, max(typical) * 0.75, f'Total: {sum(optimized)}us', fontsize=10,
            color=COLORS['success'], fontweight='bold')

    # Right: Latency distribution
    ax = axes[1]

    # Simulated latency distributions
    np.random.seed(42)
    typical_dist = np.random.lognormal(4.5, 0.5, 10000)
    optimized_dist = np.random.lognormal(3.5, 0.3, 10000)

    ax.hist(typical_dist, bins=50, alpha=0.6, label='Typical System',
            color=COLORS['warning'], density=True)
    ax.hist(optimized_dist, bins=50, alpha=0.6, label='Optimized System',
            color=COLORS['success'], density=True)

    # Add percentile lines
    for dist, color, label in [(typical_dist, COLORS['warning'], 'Typical'),
                                (optimized_dist, COLORS['success'], 'Optimized')]:
        p50 = np.percentile(dist, 50)
        p99 = np.percentile(dist, 99)
        ax.axvline(p50, color=color, linestyle='--', alpha=0.7)
        ax.axvline(p99, color=color, linestyle=':', alpha=0.7)

    ax.set_xlabel('Latency (microseconds)')
    ax.set_ylabel('Density')
    ax.set_title('Latency Distribution Comparison', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 500)

    plt.suptitle('Trading System Latency Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig_5_3_latency_breakdown')


def create_core_banking():
    """Figure 5.4: Modern Core Banking Architecture Layers."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        text_color = 'white' if color not in [COLORS['light'], COLORS['gold']] else COLORS['dark']
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9, color=text_color)
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=7, color=text_color, alpha=0.9)

    # Title
    ax.text(7, 9.5, 'Modern Core Banking Architecture', ha='center', fontsize=14, fontweight='bold')

    # Channel Layer
    ax.text(0.5, 8.7, 'Channels', fontsize=11, fontweight='bold')
    draw_box(2, 8.2, 2, 0.8, 'Mobile App', COLORS['info'])
    draw_box(4.3, 8.2, 2, 0.8, 'Web Portal', COLORS['info'])
    draw_box(6.6, 8.2, 2, 0.8, 'Branch', COLORS['info'])
    draw_box(8.9, 8.2, 2, 0.8, 'ATM', COLORS['info'])
    draw_box(11.2, 8.2, 2, 0.8, 'Partners', COLORS['info'])

    # API Gateway
    ax.text(0.5, 7.3, 'API Layer', fontsize=11, fontweight='bold')
    draw_box(2, 6.7, 5, 0.9, 'API Gateway', COLORS['primary'], 'Auth, Rate Limiting, Routing')
    draw_box(7.5, 6.7, 5, 0.9, 'Event Bus', COLORS['primary'], 'Kafka, Event Sourcing')

    # Microservices
    ax.text(0.5, 5.8, 'Services', fontsize=11, fontweight='bold')
    services = [
        ('Accounts', 1.5), ('Transfers', 3.5), ('Payments', 5.5),
        ('Cards', 7.5), ('Loans', 9.5), ('Fraud', 11.5)
    ]
    for name, x in services:
        draw_box(x, 5.2, 1.8, 0.9, name, COLORS['secondary'])

    # Core Banking Engine
    ax.text(0.5, 4.3, 'Core Engine', fontsize=11, fontweight='bold')
    draw_box(2, 3.6, 4.5, 1, 'Core Banking', COLORS['dark'], 'Account Ledger, Interest')
    draw_box(7, 3.6, 4.5, 1, 'Risk Engine', COLORS['dark'], 'Credit, AML, Fraud')

    # Data Layer
    ax.text(0.5, 2.7, 'Data', fontsize=11, fontweight='bold')
    draw_box(1.5, 2, 2.5, 0.9, 'OLTP DB', COLORS['accent'], 'PostgreSQL')
    draw_box(4.3, 2, 2.5, 0.9, 'Cache', COLORS['accent'], 'Redis Cluster')
    draw_box(7.1, 2, 2.5, 0.9, 'Search', COLORS['accent'], 'Elasticsearch')
    draw_box(9.9, 2, 2.5, 0.9, 'Analytics', COLORS['accent'], 'Snowflake')

    # Infrastructure
    ax.text(0.5, 1.1, 'Infra', fontsize=11, fontweight='bold')
    draw_box(2, 0.4, 3.5, 0.9, 'Kubernetes', COLORS['warning'], 'Container orchestration')
    draw_box(6, 0.4, 3.5, 0.9, 'Service Mesh', COLORS['warning'], 'Istio')
    draw_box(10, 0.4, 3.5, 0.9, 'Observability', COLORS['warning'], 'Prometheus + Jaeger')

    # Arrows
    for x in [3, 5.3, 7.6, 9.9, 12.2]:
        ax.annotate('', xy=(x, 8.2), xytext=(x, 7.6),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.4))

    save_figure(fig, 'fig_5_4_core_banking')


def create_risk_architecture():
    """Figure 5.5: Real-Time Risk Management Infrastructure."""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        text_color = 'white' if color not in [COLORS['light'], COLORS['gold']] else COLORS['dark']
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9, color=text_color)
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=7, color=text_color, alpha=0.9)

    # Title
    ax.text(7, 10.5, 'Real-Time Risk Management Infrastructure', ha='center', fontsize=14, fontweight='bold')

    # Input Sources
    ax.text(0.3, 9.2, 'Data Sources', fontsize=10, fontweight='bold')
    draw_box(0.3, 8.4, 2, 0.7, 'Market Data', COLORS['info'])
    draw_box(0.3, 7.6, 2, 0.7, 'Positions', COLORS['info'])
    draw_box(0.3, 6.8, 2, 0.7, 'Orders', COLORS['info'])
    draw_box(0.3, 6.0, 2, 0.7, 'Trades', COLORS['info'])

    # Pre-Trade Risk (Fast Path) - header above box
    ax.text(4.75, 9.9, 'Pre-Trade Risk', fontsize=10, fontweight='bold', ha='center', color=COLORS['success'])
    ax.text(4.75, 9.6, '(<1ms)', fontsize=9, ha='center', color=COLORS['success'])
    draw_box(3, 8.1, 3.5, 1.3, 'Pre-Trade Cache', COLORS['success'], 'Limits + Positions')

    # Real-Time Risk - header above box
    ax.text(8.75, 9.9, 'Real-Time Risk', fontsize=10, fontweight='bold', ha='center', color=COLORS['primary'])
    ax.text(8.75, 9.6, '(<100ms)', fontsize=9, ha='center', color=COLORS['primary'])
    draw_box(7, 8.1, 3.5, 1.3, 'Position Engine', COLORS['primary'], 'Real-time P&L')

    # Batch Risk - header above box
    ax.text(12.35, 9.9, 'Batch Risk', fontsize=10, fontweight='bold', ha='center', color=COLORS['warning'])
    ax.text(12.35, 9.6, '(minutes)', fontsize=9, ha='center', color=COLORS['warning'])
    draw_box(11, 8.1, 2.7, 1.3, 'VaR/Stress', COLORS['warning'], 'Monte Carlo')

    # Risk Calculations
    ax.text(3, 6.5, 'Risk Calculations', fontsize=11, fontweight='bold')
    calcs = [
        ('Limit Check', 3, 5.6, COLORS['success']),
        ('Margin Calc', 5.5, 5.6, COLORS['primary']),
        ('Greeks', 8, 5.6, COLORS['primary']),
        ('VaR', 10.5, 5.6, COLORS['warning'])
    ]
    for name, x, y, color in calcs:
        draw_box(x, y, 2.2, 0.8, name, color)

    # Compute Infrastructure
    ax.text(3, 4.6, 'Compute', fontsize=11, fontweight='bold')
    draw_box(3, 3.8, 3, 0.9, 'CPU Cluster', COLORS['dark'], 'Pre-trade, positions')
    draw_box(6.5, 3.8, 3, 0.9, 'GPU Cluster', COLORS['dark'], 'Monte Carlo, ML')
    draw_box(10, 3.8, 3, 0.9, 'FPGA', COLORS['dark'], 'Ultra-low latency')

    # Outputs
    ax.text(3, 2.8, 'Outputs', fontsize=11, fontweight='bold')
    outputs = [
        ('Alerts', 3, COLORS['accent']),
        ('Reports', 5.5, COLORS['info']),
        ('Dashboards', 8, COLORS['info']),
        ('Regulatory', 10.5, COLORS['accent'])
    ]
    for name, x, color in outputs:
        draw_box(x, 2.0, 2.2, 0.8, name, color)

    # Flow arrows from data sources
    for y in [8.7, 7.9, 7.1, 6.3]:
        ax.annotate('', xy=(3, y), xytext=(2.3, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.4))

    save_figure(fig, 'fig_5_5_risk_architecture')


if __name__ == '__main__':
    print("Generating Chapter 5 figures...")
    create_requirements_radar()
    create_compliance_architecture()
    create_latency_breakdown()
    create_core_banking()
    create_risk_architecture()
    print("Done! Generated 5 figures for Chapter 5.")
