#!/usr/bin/env python3
"""Generate figures for Chapter 13: Small Language Models for Infrastructure"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'slm': '#28A745',
    'llm': '#6F42C1',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'warning': '#FFC107',
    'edge': '#17A2B8',
    'log': '#E83E8C',
    'config': '#FD7E14'
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


def create_slm_overview():
    """Figure 13.1: SLM vs LLM Trade-offs"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Small vs Large Language Models for Infrastructure',
            fontsize=14, fontweight='bold', ha='center')

    # LLM Column
    ax.text(3.5, 7, 'Large LMs (70B+)', fontsize=11, fontweight='bold', ha='center')
    draw_box(ax, 1.5, 5.5, 4, 1.2, 'Latency: 500-2000ms', COLORS['llm'])
    draw_box(ax, 1.5, 4, 4, 1.2, 'Cost: $0.01-0.03/1K tokens', COLORS['llm'])
    draw_box(ax, 1.5, 2.5, 4, 1.2, 'Deployment: Cloud API', COLORS['llm'])
    draw_box(ax, 1.5, 1, 4, 1.2, 'Best: Complex reasoning', COLORS['llm'])

    # SLM Column
    ax.text(10.5, 7, 'Small LMs (1-7B)', fontsize=11, fontweight='bold', ha='center')
    draw_box(ax, 8.5, 5.5, 4, 1.2, 'Latency: 10-100ms', COLORS['slm'])
    draw_box(ax, 8.5, 4, 4, 1.2, 'Cost: ~$0.0001/1K tokens', COLORS['slm'])
    draw_box(ax, 8.5, 2.5, 4, 1.2, 'Deployment: Local/Edge', COLORS['slm'])
    draw_box(ax, 8.5, 1, 4, 1.2, 'Best: Real-time tasks', COLORS['slm'])

    # VS indicator
    ax.text(7, 4, 'VS', fontsize=20, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('fig_13_1_slm_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_model_selection():
    """Figure 13.2: Model Selection Guide"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Model Selection by Task Complexity',
            fontsize=14, fontweight='bold', ha='center')

    # Model tiers
    tiers = [
        ('0.5-1.5B', 'Ultra-Fast', ['Log classification', 'Simple routing', 'Status checks'], COLORS['slm']),
        ('2-3B', 'Balanced', ['Anomaly explanation', 'Config validation', 'Incident triage'], COLORS['primary']),
        ('7-8B', 'Capable', ['Complex analysis', 'Multi-step reasoning', 'Generation'], COLORS['secondary'])
    ]

    for i, (size, name, tasks, color) in enumerate(tiers):
        x = 0.5 + i * 4.5
        draw_box(ax, x, 5, 4, 2.2, f'{size}\n{name}', color)

        for j, task in enumerate(tasks):
            ax.text(x + 2, 4.5 - (j + 1) * 0.5, f'• {task}', fontsize=8, va='center')

    # Decision factors
    ax.text(7, 1.5, 'Selection Factors', fontsize=10, fontweight='bold', ha='center')
    factors = ['Latency requirement', 'Available memory', 'Task complexity', 'Context length needed']
    for i, factor in enumerate(factors):
        x = 1.5 + i * 3
        draw_box(ax, x, 0.5, 2.5, 0.8, factor, COLORS['dark'])

    plt.tight_layout()
    plt.savefig('fig_13_2_model_selection.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_finetuning():
    """Figure 13.3: Fine-Tuning Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'SLM Fine-Tuning Pipeline for Infrastructure',
            fontsize=14, fontweight='bold', ha='center')

    # Data Sources
    ax.text(1.5, 6.5, 'Data Sources', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5, 2.4, 1.2, 'Logs', COLORS['log'], 'Classified')
    draw_box(ax, 0.3, 3.5, 2.4, 1.2, 'Incidents', COLORS['primary'], 'Historical')
    draw_box(ax, 0.3, 2, 2.4, 1.2, 'Configs', COLORS['config'], 'Validated')

    # Processing
    ax.text(5, 6.5, 'Data Processing', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 3.5, 3.5, 3, 2.5, 'Dataset Builder\n\nInstruction\nInput\nOutput', COLORS['dark'])

    # Training
    ax.text(9.5, 6.5, 'LoRA Training', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 7.5, 4.5, 2, 1.5, 'Base Model', COLORS['secondary'], 'Qwen/Mistral')
    draw_box(ax, 10, 4.5, 2, 1.5, 'LoRA Adapter', COLORS['slm'], '<1% params')

    # Deployment
    ax.text(9.5, 2.5, 'Deployment', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 7.5, 1, 2, 1.2, 'Merge', COLORS['primary'])
    draw_box(ax, 10, 1, 2, 1.2, 'Quantize', COLORS['warning'])
    draw_box(ax, 12.5, 1, 1.2, 1.2, 'Deploy', COLORS['slm'])

    # Arrows
    ax.annotate('', xy=(3.4, 4.5), xytext=(2.8, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(7.4, 5.25), xytext=(6.6, 4.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(9.9, 1.6), xytext=(9.6, 1.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(12.4, 1.6), xytext=(12.1, 1.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_13_3_finetuning.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_specialized_slms():
    """Figure 13.4: Specialized SLM Architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Specialized Infrastructure SLMs',
            fontsize=14, fontweight='bold', ha='center')

    # Base model
    draw_box(ax, 5.5, 5.5, 3, 1.5, 'Base SLM\nQwen-3B / Mistral-7B', COLORS['dark'])

    # Specialized models
    specializations = [
        ('Log Analyzer', 'Classification\nPattern detection', COLORS['log'], 1),
        ('Config Advisor', 'Validation\nOptimization', COLORS['config'], 4.5),
        ('Incident Triage', 'Priority\nRouting', COLORS['primary'], 8),
        ('Runbook Matcher', 'Document\nRetrieval', COLORS['secondary'], 11.5)
    ]

    for name, desc, color, x in specializations:
        draw_box(ax, x, 2.5, 2.5, 2, f'{name}\n\n{desc}', color)

    # Arrows from base to specialized
    for _, _, _, x in specializations:
        ax.annotate('', xy=(x + 1.25, 4.6), xytext=(7, 5.4),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Edge deployment
    ax.text(7, 1.2, 'Edge Deployment: K8s Nodes, Network Devices, IoT Gateways',
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['edge'], edgecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('fig_13_4_specialized_slms.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 13 figures...")
    create_slm_overview()
    print("  Created fig_13_1_slm_overview.png")
    create_model_selection()
    print("  Created fig_13_2_model_selection.png")
    create_finetuning()
    print("  Created fig_13_3_finetuning.png")
    create_specialized_slms()
    print("  Created fig_13_4_specialized_slms.png")
    print("Done!")


if __name__ == "__main__":
    main()
