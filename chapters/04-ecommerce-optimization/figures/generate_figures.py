#!/usr/bin/env python3
"""
Generate figures for Chapter 4: E-Commerce Infrastructure Optimization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Color palette
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'info': '#0891b2',
    'light': '#f3f4f6',
    'dark': '#1f2937'
}

def save_figure(fig, name: str):
    """Save figure with consistent settings."""
    output_path = Path(__file__).parent / f"{name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_traffic_patterns():
    """Figure 4.1: E-Commerce Traffic Patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    hours = np.arange(0, 24, 0.5)

    # Normal day pattern
    ax = axes[0, 0]
    normal = 50 + 30 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 5, len(hours))
    normal = np.clip(normal, 20, 100)
    ax.fill_between(hours, normal, alpha=0.3, color=COLORS['primary'])
    ax.plot(hours, normal, color=COLORS['primary'], linewidth=2)
    ax.set_title('Normal Day Traffic', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Relative Traffic (%)')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 120)
    ax.axhline(y=100, color=COLORS['dark'], linestyle='--', alpha=0.3, label='Baseline capacity')

    # Flash sale pattern
    ax = axes[0, 1]
    flash = np.ones(len(hours)) * 30
    flash_start = np.where(hours >= 12)[0][0]
    flash_peak = np.where(hours >= 12.5)[0][0]
    flash_end = np.where(hours >= 14)[0][0]
    flash[flash_start:flash_peak] = np.linspace(30, 500, flash_peak - flash_start)
    flash[flash_peak:flash_end] = np.linspace(500, 80, flash_end - flash_peak)
    flash[flash_end:] = 80 - np.linspace(0, 50, len(hours) - flash_end)
    flash = np.clip(flash, 20, 550)
    ax.fill_between(hours, flash, alpha=0.3, color=COLORS['danger'])
    ax.plot(hours, flash, color=COLORS['danger'], linewidth=2)
    ax.axhline(y=100, color=COLORS['dark'], linestyle='--', alpha=0.3)
    ax.axvline(x=12, color=COLORS['warning'], linestyle=':', alpha=0.5, label='Sale start')
    ax.set_title('Flash Sale (100x spike)', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Relative Traffic (%)')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 600)
    ax.annotate('50x normal', xy=(12.5, 500), fontsize=9, color=COLORS['danger'])

    # Black Friday pattern
    ax = axes[1, 0]
    bf_base = 100 + 50 * np.sin((hours - 8) * np.pi / 12)
    bf = bf_base * 5 + np.random.normal(0, 20, len(hours))
    bf = np.clip(bf, 100, 800)
    ax.fill_between(hours, bf, alpha=0.3, color=COLORS['warning'])
    ax.plot(hours, bf, color=COLORS['warning'], linewidth=2)
    ax.axhline(y=100, color=COLORS['dark'], linestyle='--', alpha=0.3)
    ax.set_title('Black Friday (Sustained 5-10x)', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Relative Traffic (%)')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 900)
    ax.fill_between([9, 21], [0, 0], [900, 900], alpha=0.1, color=COLORS['warning'])
    ax.annotate('Peak shopping hours', xy=(12, 50), fontsize=9, color=COLORS['dark'])

    # Viral moment pattern
    ax = axes[1, 1]
    viral = np.ones(len(hours)) * 40
    viral_start = np.where(hours >= 15)[0][0]
    viral_peak = np.where(hours >= 15.25)[0][0]
    viral_decay = np.where(hours >= 18)[0][0]
    viral[viral_start:viral_peak] = np.exp(np.linspace(0, 5, viral_peak - viral_start)) * 40
    viral[viral_peak:viral_decay] = 300 * np.exp(-np.linspace(0, 1.5, viral_decay - viral_peak))
    viral[viral_decay:] = 60 - np.linspace(0, 20, len(hours) - viral_decay)
    viral = np.clip(viral, 30, 350)
    ax.fill_between(hours, viral, alpha=0.3, color=COLORS['secondary'])
    ax.plot(hours, viral, color=COLORS['secondary'], linewidth=2)
    ax.axhline(y=100, color=COLORS['dark'], linestyle='--', alpha=0.3)
    ax.set_title('Viral Moment (Unpredictable)', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Relative Traffic (%)')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 400)
    ax.annotate('No warning', xy=(15.3, 310), fontsize=9, color=COLORS['secondary'])
    ax.annotate('?', xy=(14.8, 50), fontsize=14, color=COLORS['secondary'], fontweight='bold')

    plt.suptitle('E-Commerce Traffic Patterns by Event Type', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig_4_1_traffic_patterns')


def create_performance_architecture():
    """Figure 4.2: Multi-Layer Performance Optimization Architecture."""
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
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    # Layer labels on left
    layers = [
        (9.2, 'Edge Layer', '1-5ms'),
        (7.2, 'CDN Layer', '5-20ms'),
        (5.2, 'Load Balancer', '1-2ms'),
        (3.2, 'Application Layer', '20-50ms'),
        (1.2, 'Data Layer', '5-30ms')
    ]

    for y, name, latency in layers:
        ax.text(0.3, y + 0.4, name, fontsize=10, fontweight='bold', va='center')
        ax.text(0.3, y, f'Budget: {latency}', fontsize=8, va='center', color=COLORS['dark'], alpha=0.7)

    # Edge Layer
    draw_box(2, 9, 2.5, 0.8, 'Edge Compute', COLORS['primary'], 'Cloudflare/Lambda@Edge')
    draw_box(5, 9, 2.5, 0.8, 'Edge Cache', COLORS['primary'], 'Static assets')
    draw_box(8, 9, 2.5, 0.8, 'WAF/DDoS', COLORS['danger'], 'Protection')
    draw_box(11, 9, 2.5, 0.8, 'Bot Detection', COLORS['warning'], 'ML-based')

    # CDN Layer
    draw_box(2, 7, 3.5, 0.8, 'Global CDN', COLORS['info'], 'Multi-region PoPs')
    draw_box(6, 7, 3.5, 0.8, 'Dynamic Caching', COLORS['info'], 'Personalized TTLs')
    draw_box(10, 7, 3.5, 0.8, 'Image Optimization', COLORS['info'], 'WebP/AVIF')

    # Load Balancer
    draw_box(3, 5, 4, 0.8, 'L7 Load Balancer', COLORS['success'], 'Header-based routing')
    draw_box(8, 5, 4, 0.8, 'Health Checks', COLORS['success'], 'Active probing')

    # Application Layer
    draw_box(1.5, 3, 2.2, 0.8, 'Web Servers', COLORS['secondary'], 'Nginx/Envoy')
    draw_box(4, 3, 2.2, 0.8, 'API Gateway', COLORS['secondary'], 'Rate limiting')
    draw_box(6.5, 3, 2.2, 0.8, 'App Servers', COLORS['secondary'], 'Business logic')
    draw_box(9, 3, 2.2, 0.8, 'ML Inference', COLORS['secondary'], 'Recommendations')
    draw_box(11.5, 3, 2.2, 0.8, 'Workers', COLORS['secondary'], 'Async jobs')

    # Data Layer
    draw_box(1.5, 1, 2, 0.8, 'Primary DB', COLORS['dark'], 'PostgreSQL')
    draw_box(4, 1, 2, 0.8, 'Read Replicas', COLORS['dark'], '3x replicas')
    draw_box(6.5, 1, 2, 0.8, 'Redis Cluster', COLORS['danger'], 'Session/Cache')
    draw_box(9, 1, 2, 0.8, 'Elasticsearch', COLORS['warning'], 'Search')
    draw_box(11.5, 1, 2, 0.8, 'Object Store', COLORS['info'], 'S3/GCS')

    # Arrows (simplified vertical flow)
    for x in [3.25, 7.75]:
        ax.annotate('', xy=(x, 8.9), xytext=(x, 7.9),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.5))

    for x in [5, 10]:
        ax.annotate('', xy=(x, 6.9), xytext=(x, 5.9),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.5))

    for x in [5, 10]:
        ax.annotate('', xy=(x, 4.9), xytext=(x, 3.9),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.5))

    for x in [2.5, 5, 7.5, 10, 12.5]:
        ax.annotate('', xy=(x, 2.9), xytext=(x, 1.9),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], alpha=0.5))

    # Title
    ax.text(7, 10.6, 'Multi-Layer E-Commerce Performance Architecture',
            ha='center', fontsize=14, fontweight='bold')
    ax.text(7, 10.2, 'Total latency budget: <100ms for interactive requests',
            ha='center', fontsize=10, color=COLORS['dark'], alpha=0.7)

    save_figure(fig, 'fig_4_2_performance_architecture')


def create_scaling_strategies():
    """Figure 4.3: Scaling Strategy Effectiveness by Event Type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Response time comparison
    ax = axes[0]
    event_types = ['Daily Peak', 'Marketing\nCampaign', 'Flash Sale', 'Viral\nMoment']
    strategies = ['Reactive', 'Scheduled', 'Predictive', 'Pre-warmed']

    data = np.array([
        [85, 70, 55, 30],   # Reactive performance
        [75, 90, 40, 25],   # Scheduled performance
        [90, 85, 80, 50],   # Predictive performance
        [95, 95, 95, 75]    # Pre-warmed performance
    ])

    x = np.arange(len(event_types))
    width = 0.2
    colors = [COLORS['danger'], COLORS['warning'], COLORS['primary'], COLORS['success']]

    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        ax.bar(x + i * width, data[i], width, label=strategy, color=color, alpha=0.8)

    ax.set_ylabel('Performance Score (0-100)')
    ax.set_title('Scaling Strategy Effectiveness', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(event_types)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color=COLORS['dark'], linestyle='--', alpha=0.3)
    ax.text(3.5, 82, 'Target', fontsize=8, color=COLORS['dark'])

    # Right: Cost vs Performance trade-off
    ax = axes[1]

    strategies_data = {
        'Static Over-provision': (95, 180, COLORS['warning']),
        'Reactive Only': (55, 85, COLORS['danger']),
        'Scheduled': (70, 95, COLORS['info']),
        'Predictive ML': (85, 105, COLORS['primary']),
        'Pre-warm + Predictive': (92, 120, COLORS['success']),
    }

    for strategy, (perf, cost, color) in strategies_data.items():
        ax.scatter(cost, perf, s=200, c=color, alpha=0.8, edgecolors='white', linewidth=2)
        ax.annotate(strategy, (cost + 3, perf), fontsize=9)

    ax.set_xlabel('Relative Cost (%)')
    ax.set_ylabel('Performance Score')
    ax.set_title('Cost vs Performance Trade-off', fontweight='bold')
    ax.set_xlim(70, 200)
    ax.set_ylim(40, 100)

    # Add optimal zone
    from matplotlib.patches import Ellipse
    optimal = Ellipse((110, 88), 40, 15, alpha=0.2, color=COLORS['success'])
    ax.add_patch(optimal)
    ax.text(110, 95, 'Optimal Zone', ha='center', fontsize=9, color=COLORS['success'])

    plt.suptitle('Scaling Strategy Analysis for E-Commerce', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig_4_3_scaling_strategies')


def create_waiting_room():
    """Figure 4.4: Virtual Waiting Room Architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9,
                color='white' if color != COLORS['light'] else COLORS['dark'])
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=7,
                    color='white' if color != COLORS['light'] else COLORS['dark'], alpha=0.9)

    def draw_arrow(start, end, label=None, color=COLORS['dark']):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
            ax.text(mid[0], mid[1] + 0.2, label, fontsize=8, ha='center', color=color)

    # Users
    ax.text(0.5, 6.5, '👥', fontsize=30, ha='center')
    ax.text(0.5, 5.8, 'Incoming\nUsers', ha='center', fontsize=9, fontweight='bold')

    # Edge/CDN
    draw_box(2, 5.5, 2, 1.2, 'Edge CDN', COLORS['primary'], 'Rate limiting')

    # Waiting Room Logic
    draw_box(5, 5.5, 3, 1.2, 'Waiting Room', COLORS['warning'], 'Queue management')

    # Queue visualization
    ax.add_patch(FancyBboxPatch((5.2, 3.5), 2.6, 1.5,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['light'], edgecolor=COLORS['dark']))
    ax.text(6.5, 4.8, 'Queue', ha='center', fontsize=9, fontweight='bold')

    # Queue items
    for i, y in enumerate([4.3, 4.0, 3.7]):
        color = COLORS['success'] if i == 0 else COLORS['warning'] if i == 1 else COLORS['info']
        ax.add_patch(plt.Circle((5.6, y), 0.12, color=color))
        ax.text(5.9, y, f'#{i+1}', fontsize=7, va='center')
        ax.text(7.2, y, f'Est: {i*15}s', fontsize=7, va='center', color=COLORS['dark'])

    # Active Session Manager
    draw_box(9.5, 5.5, 2.5, 1.2, 'Session Manager', COLORS['success'], 'Admission control')

    # Backend
    draw_box(9.5, 3.2, 2.5, 1.2, 'Backend', COLORS['secondary'], 'Protected capacity')

    # Token Store
    draw_box(5, 2, 3, 0.8, 'Token Store (Redis)', COLORS['danger'], 'Session tokens')

    # Metrics
    draw_box(9.5, 1.5, 2.5, 1, 'Metrics', COLORS['info'], 'Queue depth, wait time')

    # Arrows
    draw_arrow((1, 6.1), (2, 6.1), 'Request')
    draw_arrow((4, 6.1), (5, 6.1), 'Check capacity')
    draw_arrow((6.5, 5.5), (6.5, 5), 'Queue if full')
    draw_arrow((6.5, 3.5), (6.5, 2.8), 'Store token')
    draw_arrow((8, 6.1), (9.5, 6.1), 'Admit')
    draw_arrow((10.75, 5.5), (10.75, 4.4), 'Process')
    draw_arrow((9.5, 3.8), (8, 3.8), 'Load check', COLORS['success'])

    # Status indicators
    ax.add_patch(plt.Circle((12.8, 7), 0.15, color=COLORS['success']))
    ax.text(13.1, 7, 'Active: 8,542', fontsize=8, va='center')
    ax.add_patch(plt.Circle((12.8, 6.5), 0.15, color=COLORS['warning']))
    ax.text(13.1, 6.5, 'Queued: 45,231', fontsize=8, va='center')
    ax.add_patch(plt.Circle((12.8, 6), 0.15, color=COLORS['info']))
    ax.text(13.1, 6, 'Max: 10,000', fontsize=8, va='center')

    # Title
    ax.text(7, 7.7, 'Virtual Waiting Room Architecture for Flash Sales',
            ha='center', fontsize=14, fontweight='bold')
    ax.text(7, 7.3, 'Protects backend while ensuring fair access',
            ha='center', fontsize=10, color=COLORS['dark'], alpha=0.7)

    save_figure(fig, 'fig_4_4_waiting_room')


def create_personalization_arch():
    """Figure 4.5: Real-Time Personalization Infrastructure."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2 + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontweight='bold', fontsize=9,
                color='white' if color not in [COLORS['light'], COLORS['warning']] else COLORS['dark'])
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=7,
                    color='white' if color not in [COLORS['light'], COLORS['warning']] else COLORS['dark'])

    # Request flow (top)
    ax.text(0.5, 8, '👤', fontsize=25, ha='center')
    ax.text(0.5, 7.4, 'User', ha='center', fontsize=9)

    draw_box(2, 7.2, 2.5, 0.9, 'API Gateway', COLORS['primary'], '<1ms')

    draw_box(5.5, 7.2, 3, 0.9, 'Personalization Service', COLORS['secondary'], '50ms budget')

    # Tiered architecture
    ax.text(7, 6.7, 'Tiered Fallback System', ha='center', fontsize=11, fontweight='bold')

    # Tier 1: Real-time ML
    draw_box(1, 5, 3.5, 1.2, 'Tier 1: Real-time ML', COLORS['success'], '10-30ms | Full personalization')
    ax.text(4.7, 5.6, '→', fontsize=16)
    ax.text(4.7, 5.2, '85%', fontsize=9, color=COLORS['success'])

    # Tier 2: Segment-based
    draw_box(5.5, 5, 3.5, 1.2, 'Tier 2: Segment', COLORS['info'], '5-10ms | Cached segments')
    ax.text(9.2, 5.6, '→', fontsize=16)
    ax.text(9.2, 5.2, '12%', fontsize=9, color=COLORS['info'])

    # Tier 3: Popular items
    draw_box(10, 5, 3.5, 1.2, 'Tier 3: Popular', COLORS['warning'], '1-2ms | Static fallback')
    ax.text(11.75, 4.8, '3%', fontsize=9, color=COLORS['warning'])

    # Feature Store
    draw_box(0.5, 2.8, 3, 1.5, 'Feature Store', COLORS['danger'], 'Redis Cluster')
    ax.text(2, 2.5, '• User features', fontsize=8, ha='center')
    ax.text(2, 2.2, '• Session data', fontsize=8, ha='center')
    ax.text(2, 1.9, '• Behavioral signals', fontsize=8, ha='center')

    # ML Inference
    draw_box(4, 2.8, 3, 1.5, 'ML Inference', COLORS['secondary'], 'GPU Cluster')
    ax.text(5.5, 2.5, '• Ranking models', fontsize=8, ha='center')
    ax.text(5.5, 2.2, '• Collaborative filtering', fontsize=8, ha='center')
    ax.text(5.5, 1.9, '• Real-time scoring', fontsize=8, ha='center')

    # Product Catalog
    draw_box(7.5, 2.8, 3, 1.5, 'Product Catalog', COLORS['info'], 'Elasticsearch')
    ax.text(9, 2.5, '• 2M+ SKUs', fontsize=8, ha='center')
    ax.text(9, 2.2, '• Embeddings', fontsize=8, ha='center')
    ax.text(9, 1.9, '• Inventory status', fontsize=8, ha='center')

    # Segment Cache
    draw_box(11, 2.8, 2.5, 1.5, 'Segment Cache', COLORS['warning'])
    ax.text(12.25, 2.5, '• Pre-computed', fontsize=8, ha='center')
    ax.text(12.25, 2.2, '• 100 segments', fontsize=8, ha='center')
    ax.text(12.25, 1.9, '• Hourly refresh', fontsize=8, ha='center')

    # Arrows
    ax.annotate('', xy=(2, 7.7), xytext=(1, 7.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.annotate('', xy=(5.5, 7.7), xytext=(4.5, 7.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.annotate('', xy=(7, 7.2), xytext=(7, 6.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark']))

    # Connect tiers to data stores
    ax.annotate('', xy=(2, 4.3), xytext=(2.75, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['danger'], alpha=0.5))
    ax.annotate('', xy=(5.5, 4.3), xytext=(7.25, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], alpha=0.5))
    ax.annotate('', xy=(9, 4.3), xytext=(7.25, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['info'], alpha=0.5))
    ax.annotate('', xy=(12.25, 4.3), xytext=(11.75, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], alpha=0.5))

    # Latency budget breakdown
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 13, 1,
                boxstyle="round", facecolor=COLORS['light'], edgecolor=COLORS['dark'], alpha=0.5))
    ax.text(7, 1.2, 'Latency Budget: 50ms total', ha='center', fontsize=10, fontweight='bold')
    ax.text(3, 0.8, 'Feature fetch: 10ms', fontsize=9, ha='center')
    ax.text(7, 0.8, 'ML inference: 25ms', fontsize=9, ha='center')
    ax.text(11, 0.8, 'Network + overhead: 15ms', fontsize=9, ha='center')

    # Title
    ax.text(7, 8.7, 'Real-Time Personalization Infrastructure',
            ha='center', fontsize=14, fontweight='bold')

    save_figure(fig, 'fig_4_5_personalization_arch')


if __name__ == '__main__':
    print("Generating Chapter 4 figures...")
    create_traffic_patterns()
    create_performance_architecture()
    create_scaling_strategies()
    create_waiting_room()
    create_personalization_arch()
    print("Done! Generated 5 figures for Chapter 4.")
