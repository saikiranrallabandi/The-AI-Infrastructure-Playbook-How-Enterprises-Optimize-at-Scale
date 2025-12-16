#!/usr/bin/env python3
"""
Generate professional figures for Chapter 3: Comparative Framework
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import numpy as np
from pathlib import Path

# Professional color palette
COLORS = {
    'primary': '#0066CC',
    'secondary': '#FF6600',
    'accent1': '#00AA44',
    'accent2': '#AA0066',
    'accent3': '#6600AA',
    'light_bg': '#F5F5F5',
    'dark_text': '#333333',
    'grid': '#CCCCCC',
    'performance': '#2196F3',
    'resource': '#4CAF50',
    'operational': '#FF9800',
    'scalability': '#9C27B0',
    'economic': '#F44336',
}

def setup_style():
    """Configure matplotlib style for professional output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
    })

def fig_3_1_optimization_dimensions():
    """Create radar chart showing the 5 PROSE dimensions."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Categories
    categories = ['Performance\nEfficiency', 'Resource\nOptimization',
                  'Operational\nExcellence', 'Scalability &\nElasticity',
                  'Economic\nImpact']
    N = len(categories)

    # Angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Example data for three approaches
    traditional = [45, 40, 35, 30, 50]
    ml_based = [70, 65, 60, 75, 70]
    grpo_optimized = [90, 85, 80, 90, 85]

    # Complete the loop
    traditional += traditional[:1]
    ml_based += ml_based[:1]
    grpo_optimized += grpo_optimized[:1]

    # Plot each approach
    ax.plot(angles, traditional, 'o-', linewidth=2, label='Traditional', color=COLORS['grid'])
    ax.fill(angles, traditional, alpha=0.1, color=COLORS['grid'])

    ax.plot(angles, ml_based, 's-', linewidth=2, label='ML-Based', color=COLORS['secondary'])
    ax.fill(angles, ml_based, alpha=0.1, color=COLORS['secondary'])

    ax.plot(angles, grpo_optimized, '^-', linewidth=2, label='GRPO-Optimized', color=COLORS['primary'])
    ax.fill(angles, grpo_optimized, alpha=0.2, color=COLORS['primary'])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    # Set radial limits
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=9, color='gray')

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.title('PROSE Framework: Infrastructure Optimization Dimensions',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

def fig_3_2_utilization_curve():
    """Create utilization-performance curve showing efficiency zones."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate curve data
    utilization = np.linspace(0, 100, 1000)

    # Performance degrades exponentially as utilization increases
    # Using queuing theory inspired curve
    performance = 100 * (1 - (utilization/100)**3)
    performance = np.maximum(performance, 0)

    # Add some realistic noise smoothing
    from scipy.ndimage import gaussian_filter1d
    performance = gaussian_filter1d(performance, sigma=5)

    # Plot main curve
    ax.plot(utilization, performance, linewidth=3, color=COLORS['primary'])

    # Define zones
    # Underutilized: 0-50%
    ax.axvspan(0, 50, alpha=0.15, color='blue', label='Underutilized Zone')
    # Efficiency: 50-75%
    ax.axvspan(50, 75, alpha=0.2, color='green', label='Efficiency Zone')
    # Caution: 75-90%
    ax.axvspan(75, 90, alpha=0.15, color='orange', label='Caution Zone')
    # Danger: 90-100%
    ax.axvspan(90, 100, alpha=0.2, color='red', label='Danger Zone')

    # Add annotations
    ax.annotate('Wasted\nCapacity', xy=(25, 85), fontsize=11, ha='center',
                color=COLORS['dark_text'], fontweight='bold')
    ax.annotate('Optimal\nOperation', xy=(62.5, 75), fontsize=11, ha='center',
                color=COLORS['accent1'], fontweight='bold')
    ax.annotate('Performance\nRisk', xy=(82.5, 50), fontsize=11, ha='center',
                color=COLORS['secondary'], fontweight='bold')
    ax.annotate('Degradation', xy=(95, 20), fontsize=11, ha='center',
                color=COLORS['accent2'], fontweight='bold')

    # Add target line
    ax.axhline(y=80, color=COLORS['grid'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('Target Performance Level', xy=(5, 82), fontsize=10, color='gray')

    ax.set_xlabel('Resource Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('The Utilization-Performance Relationship', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

def fig_3_3_canary_analysis():
    """Create canary deployment analysis diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, width, height, text, color, text_color='white'):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=color, edgecolor='none',
                             mutation_scale=1)
        ax.add_patch(box)
        # Add shadow
        shadow = FancyBboxPatch((x+0.05, y-0.05), width, height,
                                boxstyle="round,pad=0.03,rounding_size=0.2",
                                facecolor='gray', edgecolor='none', alpha=0.2,
                                mutation_scale=1, zorder=-1)
        ax.add_patch(shadow)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color, wrap=True)

    def draw_arrow(start, end, color=COLORS['dark_text']):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Title
    ax.text(7, 7.5, 'Canary Deployment with Automated Analysis',
            ha='center', fontsize=16, fontweight='bold')

    # Traffic source
    draw_box(0.5, 3.5, 2, 1.5, 'Incoming\nTraffic', COLORS['dark_text'])

    # Load balancer
    draw_box(3.5, 3.5, 2, 1.5, 'Load\nBalancer', COLORS['primary'])

    # Baseline cluster (95%)
    draw_box(6.5, 5, 2.5, 1.5, 'Baseline\nCluster (95%)', COLORS['accent1'])

    # Canary cluster (5%)
    draw_box(6.5, 2, 2.5, 1.5, 'Canary\nCluster (5%)', COLORS['secondary'])

    # Metrics collector
    draw_box(10, 3.5, 2.5, 1.5, 'Metrics\nCollector', COLORS['accent3'])

    # Analysis engine
    draw_box(10, 1, 2.5, 1.5, 'Analysis\nEngine', COLORS['accent2'])

    # Decision box
    draw_box(10, 6, 2.5, 1.5, 'Promote /\nRollback', COLORS['primary'])

    # Arrows
    draw_arrow((2.5, 4.25), (3.5, 4.25))  # Traffic to LB
    draw_arrow((5.5, 4.75), (6.5, 5.5))   # LB to baseline
    draw_arrow((5.5, 3.75), (6.5, 2.75))  # LB to canary
    draw_arrow((9, 5.75), (10, 4.5))      # Baseline to metrics
    draw_arrow((9, 2.75), (10, 3.75))     # Canary to metrics
    draw_arrow((11.25, 3.5), (11.25, 2.5))  # Metrics to analysis
    draw_arrow((11.25, 2.5), (11.25, 1.75), COLORS['accent2'])  # Down arrow (implied)
    draw_arrow((11.25, 6), (11.25, 5))    # Analysis to decision

    # Add percentage labels
    ax.text(6, 5.25, '95%', fontsize=10, color=COLORS['accent1'], fontweight='bold')
    ax.text(6, 3.5, '5%', fontsize=10, color=COLORS['secondary'], fontweight='bold')

    # Add metric labels
    ax.text(10, 0.3, 'P99 Latency | Error Rate | Throughput',
            ha='left', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    return fig

def fig_3_4_rl_comparison():
    """Create RL algorithm comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Convergence comparison
    iterations = np.arange(0, 1000, 10)

    # Simulated convergence curves
    ppo_reward = 50 + 40 * (1 - np.exp(-iterations/400)) + np.random.randn(len(iterations)) * 3
    a2c_reward = 45 + 35 * (1 - np.exp(-iterations/500)) + np.random.randn(len(iterations)) * 4
    grpo_reward = 55 + 40 * (1 - np.exp(-iterations/250)) + np.random.randn(len(iterations)) * 2

    # Smooth the curves
    from scipy.ndimage import gaussian_filter1d
    ppo_reward = gaussian_filter1d(ppo_reward, sigma=3)
    a2c_reward = gaussian_filter1d(a2c_reward, sigma=3)
    grpo_reward = gaussian_filter1d(grpo_reward, sigma=3)

    ax1.plot(iterations, a2c_reward, label='A2C', linewidth=2, color=COLORS['grid'])
    ax1.plot(iterations, ppo_reward, label='PPO', linewidth=2, color=COLORS['secondary'])
    ax1.plot(iterations, grpo_reward, label='GRPO', linewidth=2.5, color=COLORS['primary'])

    ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Speed Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(30, 100)

    # Add annotation for GRPO advantage
    ax1.annotate('GRPO converges\n~40% faster', xy=(400, 90), fontsize=10,
                 ha='center', color=COLORS['primary'], fontweight='bold')

    # Right: Bar comparison
    categories = ['Sample\nEfficiency', 'Variance\nReduction', 'Production\nSafety',
                  'Implementation\nComplexity']

    x = np.arange(len(categories))
    width = 0.25

    ppo_scores = [65, 60, 70, 45]
    a2c_scores = [55, 50, 65, 55]
    grpo_scores = [90, 85, 85, 60]

    bars1 = ax2.bar(x - width, a2c_scores, width, label='A2C', color=COLORS['grid'])
    bars2 = ax2.bar(x, ppo_scores, width, label='PPO', color=COLORS['secondary'])
    bars3 = ax2.bar(x + width, grpo_scores, width, label='GRPO', color=COLORS['primary'])

    ax2.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Characteristics Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig

def fig_3_5_industry_weights():
    """Create industry weighting comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    industries = ['E-Commerce', 'Financial Services', 'Healthcare']
    dimensions = ['Performance', 'Resource Opt.', 'Operations', 'Scalability', 'Economic']

    # Weights for each industry (must sum to 100)
    ecommerce = [30, 15, 20, 25, 10]
    financial = [25, 15, 30, 15, 15]
    healthcare = [20, 15, 35, 10, 20]

    x = np.arange(len(dimensions))
    width = 0.25

    colors = [COLORS['secondary'], COLORS['primary'], COLORS['accent1']]

    bars1 = ax.bar(x - width, ecommerce, width, label='E-Commerce', color=colors[0])
    bars2 = ax.bar(x, financial, width, label='Financial Services', color=colors[1])
    bars3 = ax.bar(x + width, healthcare, width, label='Healthcare', color=colors[2])

    ax.set_ylabel('Weight (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('PROSE Dimension', fontsize=12, fontweight='bold')
    ax.set_title('Industry-Specific PROSE Dimension Weights', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 40)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add annotations for key insights
    ax.annotate('E-Commerce prioritizes\nscalability for flash sales',
                xy=(3, 27), fontsize=9, ha='center', color=colors[0],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.annotate('Financial services\nfocus on operations\n(compliance)',
                xy=(2, 33), fontsize=9, ha='center', color=colors[1],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

def main():
    """Generate all figures for Chapter 3."""
    setup_style()

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ('fig_3_1_optimization_dimensions', fig_3_1_optimization_dimensions),
        ('fig_3_2_utilization_curve', fig_3_2_utilization_curve),
        ('fig_3_3_canary_analysis', fig_3_3_canary_analysis),
        ('fig_3_4_rl_comparison', fig_3_4_rl_comparison),
        ('fig_3_5_industry_weights', fig_3_5_industry_weights),
    ]

    for name, func in figures:
        print(f"Generating {name}...")
        fig = func()

        # Save as PNG (high DPI for print)
        fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # Save as SVG (vector format)
        fig.savefig(output_dir / f"{name}.svg", bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        plt.close(fig)
        print(f"  Saved {name}.png and {name}.svg")

    print("\nAll Chapter 3 figures generated successfully!")

if __name__ == "__main__":
    main()
