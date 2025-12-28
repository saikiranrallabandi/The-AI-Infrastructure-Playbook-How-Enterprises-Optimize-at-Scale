#!/usr/bin/env python3
"""Generate figures for Chapter 18: Security and Compliance Automation"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'security': '#DC3545',
    'compliance': '#28A745',
    'policy': '#6F42C1',
    'primary': '#4A90D9',
    'secondary': '#7B68EE',
    'dark': '#343A40',
    'warning': '#FFC107',
    'threat': '#E83E8C'
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


def create_security_overview():
    """Figure 18.1: Security Integration Overview"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Security-Integrated Infrastructure Optimization',
            fontsize=14, fontweight='bold', ha='center')

    # Optimization layer
    ax.text(3, 6.8, 'Optimization', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.5, 5, 2, 1.5, 'Cost\nOptimizer', COLORS['primary'])
    draw_box(ax, 2.8, 5, 2, 1.5, 'Resource\nScaler', COLORS['primary'])

    # Security layer
    ax.text(8, 6.8, 'Security Controls', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 5.5, 5, 2, 1.5, 'Policy\nEngine', COLORS['policy'])
    draw_box(ax, 7.8, 5, 2, 1.5, 'Threat\nDetection', COLORS['threat'])
    draw_box(ax, 10.1, 5, 2, 1.5, 'Compliance\nMonitor', COLORS['compliance'])

    # Integration
    draw_box(ax, 3, 2.5, 8, 2, 'Secure AIOps Platform\n\nActions validated against policies before execution', COLORS['dark'])

    # Arrows
    for x in [1.5, 3.8, 6.5, 8.8, 11.1]:
        ax.annotate('', xy=(7, 4.6), xytext=(x, 4.9),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_18_1_security_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_policy_flow():
    """Figure 18.2: Policy Evaluation Flow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Policy-as-Code Evaluation Flow',
            fontsize=14, fontweight='bold', ha='center')

    # Flow stages
    draw_box(ax, 0.5, 4, 2.5, 2.5, 'Resource\nChange\n\nCreate/Update', COLORS['primary'])

    draw_box(ax, 3.5, 4, 2.5, 2.5, 'Policy\nEngine\n\nOPA/Rego', COLORS['policy'])

    # Decision branches
    draw_box(ax, 6.5, 5.5, 2, 1.2, 'ALLOW', COLORS['compliance'])
    draw_box(ax, 6.5, 4, 2, 1.2, 'DENY', COLORS['security'])
    draw_box(ax, 6.5, 2.5, 2, 1.2, 'WARN', COLORS['warning'])

    # Outcomes
    draw_box(ax, 9, 5.5, 2, 1.2, 'Proceed', COLORS['compliance'])
    draw_box(ax, 9, 4, 2, 1.2, 'Block', COLORS['security'])
    draw_box(ax, 9, 2.5, 2, 1.2, 'Log + Alert', COLORS['warning'])

    # Audit
    draw_box(ax, 11.5, 3.5, 2.2, 2.5, 'Audit\nLog\n\nCompliance\nEvidence', COLORS['dark'])

    # Arrows
    ax.annotate('', xy=(3.4, 5.25), xytext=(3.1, 5.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6.4, 5.75), xytext=(6.1, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_18_2_policy_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_threat_detection():
    """Figure 18.3: Threat Detection Pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'AI-Powered Threat Detection Pipeline',
            fontsize=14, fontweight='bold', ha='center')

    # Data sources
    ax.text(1.5, 6.5, 'Data Sources', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 0.3, 5, 2.4, 1, 'Access Logs', COLORS['primary'])
    draw_box(ax, 0.3, 3.7, 2.4, 1, 'Network Flow', COLORS['primary'])
    draw_box(ax, 0.3, 2.4, 2.4, 1, 'API Calls', COLORS['primary'])

    # Feature extraction
    draw_box(ax, 3.2, 3, 2.5, 3.5, 'Feature\nExtraction\n\nAccess\nNetwork\nResource', COLORS['secondary'])

    # ML Detection
    draw_box(ax, 6.2, 4, 2.5, 2.5, 'Anomaly\nDetection\n\nIsolation\nForest', COLORS['threat'])

    # Threat Intel
    draw_box(ax, 6.2, 1, 2.5, 1.5, 'Threat Intel\nIntegration', COLORS['dark'])

    # Alert
    draw_box(ax, 9.2, 3.5, 2.5, 2.5, 'Alert\nCorrelation\n\nDedupe\nEnrich', COLORS['warning'])

    # Response
    draw_box(ax, 12, 3.5, 1.7, 2.5, 'Response\n\nAuto\nManual', COLORS['security'])

    # Arrows
    ax.annotate('', xy=(3.1, 4.5), xytext=(2.8, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig('fig_18_3_threat_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_compliance_frameworks():
    """Figure 18.4: Compliance Framework Mapping"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.7, 'Multi-Framework Compliance Mapping',
            fontsize=14, fontweight='bold', ha='center')

    # Frameworks
    ax.text(2, 6.5, 'Frameworks', fontsize=10, fontweight='bold', ha='center')
    frameworks = [('SOC 2', 0.5), ('PCI DSS', 2.5), ('HIPAA', 0.5), ('ISO 27001', 2.5)]
    for i, (name, offset) in enumerate(frameworks):
        y = 5 if i < 2 else 3.5
        draw_box(ax, offset, y, 1.8, 1.2, name, COLORS['compliance'])

    # Policy mapping
    draw_box(ax, 5, 3.5, 3, 3, 'Policy\nMapping\n\nControl to Policy\nMapping', COLORS['policy'])

    # Policies
    ax.text(11, 6.5, 'Policies', fontsize=10, fontweight='bold', ha='center')
    policies = ['SEC-001', 'SEC-002', 'SEC-003', 'SEC-004']
    for i, policy in enumerate(policies):
        y = 5.2 - i * 0.9
        draw_box(ax, 9, y, 1.8, 0.7, policy, COLORS['primary'])

    # Report
    draw_box(ax, 11.5, 3, 2.2, 3.5, 'Compliance\nReport\n\nDashboard\nEvidence\nAudit', COLORS['dark'])

    # Arrows
    ax.annotate('', xy=(4.9, 5), xytext=(4.4, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(8.9, 5), xytext=(8.1, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(11.4, 4.5), xytext=(10.9, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_18_4_compliance_frameworks.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating Chapter 18 figures...")
    create_security_overview()
    print("  Created fig_18_1_security_overview.png")
    create_policy_flow()
    print("  Created fig_18_2_policy_flow.png")
    create_threat_detection()
    print("  Created fig_18_3_threat_detection.png")
    create_compliance_frameworks()
    print("  Created fig_18_4_compliance_frameworks.png")
    print("Done!")


if __name__ == "__main__":
    main()
