#!/usr/bin/env python3
"""
Generate figures for Chapter 6: Healthcare Infrastructure Optimization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'

# Color palette - Healthcare focused
COLORS = {
    'primary': '#2E86AB',      # Healthcare blue
    'secondary': '#A23B72',     # Magenta for alerts
    'success': '#28A745',       # Green
    'warning': '#FFC107',       # Yellow/amber
    'danger': '#DC3545',        # Red
    'info': '#17A2B8',          # Cyan
    'dark': '#343A40',          # Dark gray
    'light': '#F8F9FA',         # Light gray
    'ehr': '#4A90D9',           # EHR blue
    'imaging': '#7B68EE',       # Imaging purple
    'telemedicine': '#20B2AA',  # Telemedicine teal
    'cds': '#FF6B6B',           # CDS coral
    'compliance': '#2ECC71'     # Compliance green
}


def create_requirements_radar():
    """Figure 6.1: Healthcare Infrastructure Requirements Radar"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Categories
    categories = [
        'Patient Safety',
        'HIPAA Compliance',
        'Availability',
        'Interoperability',
        'Performance',
        'Scalability',
        'Security',
        'Clinical Workflow'
    ]

    # Values for different healthcare domains
    domains = {
        'EHR Systems': [95, 95, 99, 90, 80, 85, 95, 95],
        'Medical Imaging': [90, 90, 95, 85, 85, 90, 90, 80],
        'Telemedicine': [85, 90, 95, 80, 90, 85, 90, 85],
        'Lab Systems': [95, 90, 99, 90, 85, 80, 90, 85]
    }

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    colors = [COLORS['ehr'], COLORS['imaging'], COLORS['telemedicine'], COLORS['warning']]

    for i, (domain, values) in enumerate(domains.items()):
        values = values + values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=domain, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.title('Healthcare Infrastructure Requirements by Domain', size=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig('fig_6_1_requirements_radar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_compliance_architecture():
    """Figure 6.2: Healthcare Compliance Architecture"""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, width, height, label, color, sublabel=None):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2 + (0.15 if sublabel else 0), label,
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=8, color='white', alpha=0.9)

    # Title
    ax.text(7.5, 9.7, 'Healthcare Compliance and Security Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Security Perimeter
    perimeter = FancyBboxPatch(
        (0.5, 0.5), 14, 8.5,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='none', edgecolor=COLORS['danger'], linewidth=3, linestyle='--'
    )
    ax.add_patch(perimeter)
    ax.text(7.5, 9.2, 'Zero-Trust Security Perimeter', ha='center',
            fontsize=11, color=COLORS['danger'], fontweight='bold')

    # Access Control Layer
    ax.text(2.5, 8.5, 'Access Control', fontsize=10, fontweight='bold')
    draw_box(1, 7.5, 2, 0.8, 'IAM', COLORS['primary'], 'Identity Mgmt')
    draw_box(3.5, 7.5, 2, 0.8, 'RBAC', COLORS['primary'], 'Role-Based')
    draw_box(6, 7.5, 2, 0.8, 'MFA', COLORS['primary'], 'Multi-Factor')
    draw_box(8.5, 7.5, 2.5, 0.8, 'Break Glass', COLORS['warning'], 'Emergency')

    # PHI Data Protection Layer
    ax.text(2.5, 6.8, 'PHI Data Protection', fontsize=10, fontweight='bold')
    draw_box(1, 5.8, 2.5, 0.8, 'Encryption', COLORS['compliance'], 'AES-256')
    draw_box(4, 5.8, 2.5, 0.8, 'Tokenization', COLORS['compliance'], 'De-identify')
    draw_box(7, 5.8, 2.5, 0.8, 'Masking', COLORS['compliance'], 'Display')
    draw_box(10, 5.8, 2.5, 0.8, 'Key Mgmt', COLORS['compliance'], 'HSM')

    # Audit & Monitoring Layer
    ax.text(2.5, 5.1, 'Audit & Monitoring', fontsize=10, fontweight='bold')
    draw_box(1, 4.1, 2.5, 0.8, 'Access Logs', COLORS['info'], 'PHI Access')
    draw_box(4, 4.1, 2.5, 0.8, 'SIEM', COLORS['info'], 'Security Events')
    draw_box(7, 4.1, 2.5, 0.8, 'Anomaly Det.', COLORS['info'], 'ML-Based')
    draw_box(10, 4.1, 2.5, 0.8, 'Alerts', COLORS['danger'], 'Real-time')

    # Data Flow Layer
    ax.text(2.5, 3.4, 'Protected Data Flow', fontsize=10, fontweight='bold')
    draw_box(1, 2.4, 3, 0.8, 'EHR Data', COLORS['ehr'], 'Patient Records')
    draw_box(4.5, 2.4, 3, 0.8, 'Imaging', COLORS['imaging'], 'DICOM')
    draw_box(8, 2.4, 3, 0.8, 'Lab Results', COLORS['warning'], 'HL7/FHIR')

    # Compliance Reporting
    ax.text(2.5, 1.7, 'Compliance', fontsize=10, fontweight='bold')
    draw_box(1, 0.7, 3, 0.8, 'HIPAA Reports', COLORS['dark'], 'Periodic')
    draw_box(4.5, 0.7, 3, 0.8, 'Risk Assess.', COLORS['dark'], 'Continuous')
    draw_box(8, 0.7, 3, 0.8, 'Breach Notify', COLORS['danger'], '< 60 days')

    # Compliance badges on the right
    badges = [
        ('HIPAA', 13.5, 7.5),
        ('HITECH', 13.5, 6.5),
        ('SOC 2', 13.5, 5.5),
        ('HITRUST', 13.5, 4.5)
    ]

    for badge, bx, by in badges:
        circle = Circle((bx, by), 0.45, facecolor=COLORS['success'],
                        edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(bx, by, badge, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('fig_6_2_compliance_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_ehr_architecture():
    """Figure 6.3: EHR System Architecture"""
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
        ax.text(x + width/2, y + height/2 + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    # Title
    ax.text(7, 9.7, 'Modern EHR System Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Clinical Users Layer
    ax.text(0.3, 9.2, 'Clinical Users', fontsize=10, fontweight='bold')
    draw_box(0.5, 8.4, 2, 0.7, 'Physicians', COLORS['primary'])
    draw_box(2.8, 8.4, 2, 0.7, 'Nurses', COLORS['primary'])
    draw_box(5.1, 8.4, 2, 0.7, 'Pharmacists', COLORS['primary'])
    draw_box(7.4, 8.4, 2, 0.7, 'Lab Techs', COLORS['primary'])
    draw_box(9.7, 8.4, 2, 0.7, 'Admins', COLORS['primary'])
    draw_box(12, 8.4, 1.5, 0.7, 'Patients', COLORS['info'])

    # Presentation Layer
    ax.text(0.3, 7.8, 'Presentation', fontsize=10, fontweight='bold')
    draw_box(0.5, 6.9, 2.5, 0.7, 'Web App', COLORS['ehr'], 'React')
    draw_box(3.3, 6.9, 2.5, 0.7, 'Mobile', COLORS['ehr'], 'Native')
    draw_box(6.1, 6.9, 2.5, 0.7, 'Voice UI', COLORS['ehr'], 'Ambient')
    draw_box(8.9, 6.9, 2.5, 0.7, 'Patient Portal', COLORS['info'], 'Self-Service')
    draw_box(11.7, 6.9, 1.8, 0.7, 'API', COLORS['dark'], 'SMART')

    # API Gateway Layer
    ax.text(0.3, 6.3, 'API Gateway', fontsize=10, fontweight='bold')
    gateway = FancyBboxPatch(
        (0.5, 5.5), 13, 0.7,
        boxstyle="round,pad=0.02",
        facecolor=COLORS['warning'], edgecolor='white', linewidth=2
    )
    ax.add_patch(gateway)
    ax.text(7, 5.85, 'API Gateway: Authentication | Rate Limiting | Routing | FHIR Compliance',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Clinical Services Layer
    ax.text(0.3, 5.0, 'Clinical Services', fontsize=10, fontweight='bold')
    draw_box(0.5, 4.1, 2, 0.8, 'Patient\nMgmt', COLORS['ehr'])
    draw_box(2.8, 4.1, 2, 0.8, 'Orders\n(CPOE)', COLORS['ehr'])
    draw_box(5.1, 4.1, 2, 0.8, 'Clinical\nNotes', COLORS['ehr'])
    draw_box(7.4, 4.1, 2, 0.8, 'Medication\nAdmin', COLORS['ehr'])
    draw_box(9.7, 4.1, 2, 0.8, 'Results\nReview', COLORS['ehr'])
    draw_box(12, 4.1, 1.5, 0.8, 'CDS', COLORS['cds'])

    # Integration Layer
    ax.text(0.3, 3.5, 'Integration', fontsize=10, fontweight='bold')
    draw_box(0.5, 2.6, 2.5, 0.7, 'HL7 v2', COLORS['info'], 'Legacy')
    draw_box(3.3, 2.6, 2.5, 0.7, 'FHIR R4', COLORS['compliance'], 'Modern')
    draw_box(6.1, 2.6, 2.5, 0.7, 'DICOM', COLORS['imaging'], 'Imaging')
    draw_box(8.9, 2.6, 2.5, 0.7, 'X12/EDI', COLORS['dark'], 'Claims')
    draw_box(11.7, 2.6, 1.8, 0.7, 'CDA', COLORS['dark'], 'Docs')

    # Data Layer
    ax.text(0.3, 2.0, 'Data Layer', fontsize=10, fontweight='bold')
    draw_box(0.5, 1.1, 3, 0.7, 'Patient DB', COLORS['primary'], 'PostgreSQL')
    draw_box(3.8, 1.1, 3, 0.7, 'Clinical Data', COLORS['primary'], 'Oracle')
    draw_box(7.1, 1.1, 3, 0.7, 'Doc Store', COLORS['primary'], 'MongoDB')
    draw_box(10.4, 1.1, 3, 0.7, 'Analytics', COLORS['primary'], 'Snowflake')

    # Arrows connecting layers
    arrow_style = dict(arrowstyle='->', color='gray', lw=1.5)
    for x in [1.5, 3.8, 6.2, 8.5, 10.7]:
        ax.annotate('', xy=(x, 6.9), xytext=(x, 8.4),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()
    plt.savefig('fig_6_3_ehr_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_imaging_infrastructure():
    """Figure 6.4: Medical Imaging AI Infrastructure"""
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
        ax.text(x + width/2, y + height/2 + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    # Title
    ax.text(7, 9.7, 'Medical Imaging AI Infrastructure',
            fontsize=14, fontweight='bold', ha='center')

    # Imaging Modalities
    ax.text(0.3, 9.2, 'Modalities', fontsize=10, fontweight='bold')
    draw_box(0.5, 8.3, 1.8, 0.7, 'CT', COLORS['imaging'])
    draw_box(2.6, 8.3, 1.8, 0.7, 'MRI', COLORS['imaging'])
    draw_box(4.7, 8.3, 1.8, 0.7, 'X-Ray', COLORS['imaging'])
    draw_box(6.8, 8.3, 1.8, 0.7, 'US', COLORS['imaging'])
    draw_box(8.9, 8.3, 1.8, 0.7, 'Mammo', COLORS['imaging'])
    draw_box(11, 8.3, 2.5, 0.7, 'Pathology', COLORS['imaging'], 'WSI')

    # PACS Layer
    ax.text(0.3, 7.7, 'PACS', fontsize=10, fontweight='bold')
    pacs_box = FancyBboxPatch(
        (0.5, 6.7), 13, 0.8,
        boxstyle="round,pad=0.02",
        facecolor=COLORS['primary'], edgecolor='white', linewidth=2
    )
    ax.add_patch(pacs_box)
    ax.text(7, 7.1, 'PACS: DICOM Archive | Query/Retrieve | Worklist | Routing',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # AI Processing Layer
    ax.text(0.3, 6.1, 'AI Processing', fontsize=10, fontweight='bold')
    draw_box(0.5, 5.1, 2.5, 0.8, 'Pre-process', COLORS['dark'], 'Normalization')
    draw_box(3.3, 5.1, 2.5, 0.8, 'Segmentation', COLORS['cds'], 'U-Net')
    draw_box(6.1, 5.1, 2.5, 0.8, 'Detection', COLORS['cds'], 'YOLO/Faster-RCNN')
    draw_box(8.9, 5.1, 2.5, 0.8, 'Classification', COLORS['cds'], 'ResNet/ViT')
    draw_box(11.7, 5.1, 1.8, 0.8, 'Report Gen', COLORS['cds'])

    # GPU Infrastructure
    ax.text(0.3, 4.5, 'Compute', fontsize=10, fontweight='bold')
    draw_box(0.5, 3.5, 3.5, 0.8, 'GPU Cluster', COLORS['warning'], 'NVIDIA A100')
    draw_box(4.3, 3.5, 3.5, 0.8, 'Inference Servers', COLORS['warning'], 'Triton')
    draw_box(8.1, 3.5, 2.5, 0.8, 'Model Registry', COLORS['info'])
    draw_box(10.9, 3.5, 2.6, 0.8, 'MLOps', COLORS['info'], 'Kubeflow')

    # Tiered Storage
    ax.text(0.3, 2.9, 'Storage Tiers', fontsize=10, fontweight='bold')
    draw_box(0.5, 1.9, 2.5, 0.8, 'Hot (SSD)', COLORS['danger'], '< 30 days')
    draw_box(3.3, 1.9, 2.5, 0.8, 'Warm (HDD)', COLORS['warning'], '30-365 days')
    draw_box(6.1, 1.9, 2.5, 0.8, 'Cold (S3)', COLORS['info'], '1-7 years')
    draw_box(8.9, 1.9, 2.5, 0.8, 'Archive', COLORS['dark'], '7+ years')
    draw_box(11.7, 1.9, 1.8, 0.8, 'VNA', COLORS['primary'])

    # Output Layer
    ax.text(0.3, 1.3, 'Outputs', fontsize=10, fontweight='bold')
    draw_box(0.5, 0.3, 2.5, 0.8, 'DICOM SR', COLORS['compliance'], 'Structured')
    draw_box(3.3, 0.3, 2.5, 0.8, 'Viewer', COLORS['compliance'], 'Web/3D')
    draw_box(6.1, 0.3, 2.5, 0.8, 'Worklist', COLORS['compliance'], 'Prioritized')
    draw_box(8.9, 0.3, 2.5, 0.8, 'EHR', COLORS['ehr'], 'Integration')
    draw_box(11.7, 0.3, 1.8, 0.8, 'Alerts', COLORS['danger'])

    # Performance annotations
    ax.text(13.8, 7.1, '<3s\nfirst\nimage', ha='center', va='center',
            fontsize=8, color=COLORS['primary'], fontweight='bold')
    ax.text(13.8, 5.5, '<60s\nAI\nanalysis', ha='center', va='center',
            fontsize=8, color=COLORS['cds'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_6_4_imaging_infrastructure.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_cds_architecture():
    """Figure 6.5: Clinical Decision Support Architecture"""
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
        ax.text(x + width/2, y + height/2 + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    # Title
    ax.text(7, 9.7, 'Clinical Decision Support System Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Input Sources
    ax.text(0.3, 9.2, 'Clinical Triggers', fontsize=10, fontweight='bold')
    draw_box(0.5, 8.3, 2.5, 0.7, 'Order Entry', COLORS['ehr'], 'CPOE')
    draw_box(3.3, 8.3, 2.5, 0.7, 'Medication', COLORS['ehr'], 'Prescribing')
    draw_box(6.1, 8.3, 2.5, 0.7, 'Lab Results', COLORS['warning'], 'New Values')
    draw_box(8.9, 8.3, 2.5, 0.7, 'Documentation', COLORS['ehr'], 'Notes')
    draw_box(11.7, 8.3, 1.8, 0.7, 'Vitals', COLORS['info'])

    # Context Assembly
    ax.text(0.3, 7.7, 'Context Assembly', fontsize=10, fontweight='bold')
    draw_box(0.5, 6.7, 3, 0.8, 'Patient Context', COLORS['primary'], 'Demographics, Dx')
    draw_box(3.8, 6.7, 3, 0.8, 'Active Meds', COLORS['primary'], 'Current Rx')
    draw_box(7.1, 6.7, 3, 0.8, 'Allergies', COLORS['danger'], 'Documented')
    draw_box(10.4, 6.7, 3, 0.8, 'Lab Values', COLORS['primary'], 'Recent')

    # CDS Engine
    ax.text(0.3, 6.1, 'CDS Engine', fontsize=10, fontweight='bold')
    engine_box = FancyBboxPatch(
        (0.5, 4.4), 13, 1.5,
        boxstyle="round,pad=0.03",
        facecolor=COLORS['cds'], edgecolor='white', linewidth=3
    )
    ax.add_patch(engine_box)

    # Engine components inside
    draw_box(0.8, 4.7, 2.5, 0.9, 'Drug-Drug\nInteractions', COLORS['danger'])
    draw_box(3.6, 4.7, 2.5, 0.9, 'Drug-Allergy\nChecks', COLORS['danger'])
    draw_box(6.4, 4.7, 2.5, 0.9, 'Dose Range\nValidation', COLORS['warning'])
    draw_box(9.2, 4.7, 2.2, 0.9, 'Guidelines\nEngine', COLORS['compliance'])
    draw_box(11.7, 4.7, 1.5, 0.9, 'ML\nModels', COLORS['info'])

    # Knowledge Base
    ax.text(0.3, 3.8, 'Knowledge Base', fontsize=10, fontweight='bold')
    draw_box(0.5, 2.9, 2.5, 0.7, 'Drug DB', COLORS['dark'], 'FDB/Medi-Span')
    draw_box(3.3, 2.9, 2.5, 0.7, 'Guidelines', COLORS['dark'], 'Clinical')
    draw_box(6.1, 2.9, 2.5, 0.7, 'Evidence', COLORS['dark'], 'Literature')
    draw_box(8.9, 2.9, 2.5, 0.7, 'Local Rules', COLORS['dark'], 'Institution')
    draw_box(11.7, 2.9, 1.8, 0.7, 'ML Models', COLORS['dark'])

    # Alert Delivery
    ax.text(0.3, 2.3, 'Alert Delivery', fontsize=10, fontweight='bold')
    draw_box(0.5, 1.3, 3, 0.8, 'Interruptive', COLORS['danger'], 'Hard Stop')
    draw_box(3.8, 1.3, 3, 0.8, 'Non-Interruptive', COLORS['warning'], 'Info Panel')
    draw_box(7.1, 1.3, 3, 0.8, 'Suggestions', COLORS['compliance'], 'Order Sets')
    draw_box(10.4, 1.3, 3, 0.8, 'Documentation', COLORS['info'], 'Audit Trail')

    # Feedback Loop
    ax.text(0.3, 0.7, 'Feedback', fontsize=10, fontweight='bold')
    draw_box(0.5, 0.2, 3, 0.4, 'Override Tracking', COLORS['primary'])
    draw_box(3.8, 0.2, 3, 0.4, 'Outcome Analysis', COLORS['primary'])
    draw_box(7.1, 0.2, 3, 0.4, 'Alert Fatigue Metrics', COLORS['primary'])
    draw_box(10.4, 0.2, 3, 0.4, 'Rule Refinement', COLORS['primary'])

    # Alert severity legend
    ax.text(13.5, 5.8, 'Alert\nSeverity', ha='center', fontsize=8, fontweight='bold')
    for i, (level, color) in enumerate([('High', COLORS['danger']),
                                         ('Medium', COLORS['warning']),
                                         ('Low', COLORS['info'])]):
        y_pos = 5.4 - i * 0.4
        circle = Circle((13.2, y_pos), 0.15, facecolor=color, edgecolor='white')
        ax.add_patch(circle)
        ax.text(13.5, y_pos, level, va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig('fig_6_5_cds_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Generate all figures for Chapter 6."""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Generating Chapter 6 figures...")

    print("  Creating Figure 6.1: Healthcare Requirements Radar...")
    create_requirements_radar()

    print("  Creating Figure 6.2: Healthcare Compliance Architecture...")
    create_compliance_architecture()

    print("  Creating Figure 6.3: EHR System Architecture...")
    create_ehr_architecture()

    print("  Creating Figure 6.4: Medical Imaging Infrastructure...")
    create_imaging_infrastructure()

    print("  Creating Figure 6.5: Clinical Decision Support Architecture...")
    create_cds_architecture()

    print("Done! All figures generated successfully.")


if __name__ == "__main__":
    main()
