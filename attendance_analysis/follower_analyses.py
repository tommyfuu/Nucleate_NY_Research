import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
### analysis 3. Industry tab - PIE CHART VERSION
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_industry = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_followers_past 365 days.xls',
    sheet_name='Industry'
)

# --- REVISED SECTOR MAPPING ---
industry_to_sector = {
    # # BIOTECH, PHARMA & HEALTHCARE
    # 'Biotechnology Research': 'Biotech, Pharma & Healthcare',
    # 'Pharmaceutical Manufacturing': 'Biotech, Pharma & Healthcare',
    # 'Hospitals': 'Biotech, Pharma & Healthcare',
    # 'Research Services': 'Biotech, Pharma & Healthcare',
    # 'Hospitals and Health Care': 'Biotech, Pharma & Healthcare',
    # 'Medical Equipment Manufacturing': 'Biotech, Pharma & Healthcare',
    # 'Medical and Diagnostic Laboratories': 'Biotech, Pharma & Healthcare',
    # 'Medical Practices': 'Biotech, Pharma & Healthcare',
    # 'Wellness and Fitness Services': 'Biotech, Pharma & Healthcare',
    # 'Public Health': 'Biotech, Pharma & Healthcare',
    # 'Health and Human Services': 'Biotech, Pharma & Healthcare',
    # 'Mental Health Care': 'Biotech, Pharma & Healthcare',
    # 'Outpatient Care Centers': 'Biotech, Pharma & Healthcare',
    # 'Personal Care Product Manufacturing': 'Biotech, Pharma & Healthcare',
    # 'Wholesale Drugs and Sundries': 'Biotech, Pharma & Healthcare',
    # 'Dentists': 'Biotech, Pharma & Healthcare',

    # BIOMEDICAL RESEARCH
    'Biotechnology Research': 'Biomedical Research',
    'Research Services': 'Biomedical Research',
    'Medical and Diagnostic Laboratories': 'Biomedical Research',
    
    # BIOTECH & HEALTH TECH
    'Medical Equipment Manufacturing': 'Biotech & Health Tech',
    'Wellness and Fitness Services': 'Biotech & Health Tech',
    'Personal Care Product Manufacturing': 'Biotech & Health Tech',
    
    # PHARMA
    'Wholesale Drugs and Sundries': 'Pharma',
    'Pharmaceutical Manufacturing': 'Pharma',
    
    # HOSPITAL SERVICES
    'Hospitals': 'Hospital Services',
    'Hospitals and Health Care': 'Hospital Services',
    'Medical Practices': 'Hospital Services',
    'Public Health': 'Hospital Services',
    'Health and Human Services': 'Hospital Services',
    'Mental Health Care': 'Hospital Services',
    'Outpatient Care Centers': 'Hospital Services',
    'Dentists': 'Hospital Services',

    # VENTURE CAPITAL & FINANCE
    'Venture Capital and Private Equity Principals': 'VC & Finance',
    'Investment Management': 'VC & Finance',
    'Financial Services': 'VC & Finance',
    'Capital Markets': 'VC & Finance',
    'Banking': 'VC & Finance',
    'Investment Banking': 'VC & Finance',
    'Accounting': 'VC & Finance',
    'Credit Intermediation': 'VC & Finance',
    'Insurance': 'VC & Finance',
    'Insurance Carriers': 'VC & Finance',
    'Trusts and Estates': 'VC & Finance',

    # ACADEMIA & EDUCATION
    'Higher Education': 'Academia & Education',
    'Education': 'Academia & Education',
    'Education Administration Programs': 'Academia & Education',
    'Primary and Secondary Education': 'Academia & Education',
    'Professional Training and Coaching': 'Academia & Education',
    'E-Learning Providers': 'Academia & Education',

    # TECHNOLOGY & SOFTWARE
    'Software Development': 'Technology & Software',
    'IT Services and IT Consulting': 'Technology & Software',
    'Technology, Information and Internet': 'Technology & Software',
    'Technology, Information and Media': 'Technology & Software',
    'Computer and Network Security': 'Technology & Software',
    'Internet Marketplace Platforms': 'Technology & Software',
    'IT System Custom Software Development': 'Technology & Software',
    'Nanotechnology Research': 'Technology & Software',
    'Semiconductor Manufacturing': 'Technology & Software',
    'Telecommunications': 'Technology & Software',
    'Social Networking Platforms': 'Technology & Software',
    'Internet Publishing': 'Technology & Software',
    'Online Audio and Video Media': 'Technology & Software',
    'Computer Hardware Manufacturing': 'Technology & Software',
    'Computer Networking Products': 'Technology & Software',

    # PROFESSIONAL SERVICES & LEGAL
    'Business Consulting and Services': 'Professional Services & Legal',
    'Law Practice': 'Professional Services & Legal',
    'Legal Services': 'Professional Services & Legal',
    'Staffing and Recruiting': 'Professional Services & Legal',
    'Executive Search Services': 'Professional Services & Legal',
    'Strategic Management Services': 'Professional Services & Legal',
    'Human Resources Services': 'Professional Services & Legal',
    'Advertising Services': 'Professional Services & Legal',
    'Public Relations and Communications Services': 'Professional Services & Legal',
    'Market Research': 'Professional Services & Legal',
    'Marketing Services': 'Professional Services & Legal',
    'Design Services': 'Professional Services & Legal',
    'Writing and Editing': 'Professional Services & Legal',

    # NON-PROFIT, GOV & POLICY
    'Non-profit Organizations': 'Non-profit, Gov & Policy',
    'Civic and Social Organizations': 'Non-profit, Gov & Policy',
    'Government Administration': 'Non-profit, Gov & Policy',
    'Public Policy Offices': 'Non-profit, Gov & Policy',
    'Think Tanks': 'Non-profit, Gov & Policy',
    'International Affairs': 'Non-profit, Gov & Policy',
    'Environmental Services': 'Non-profit, Gov & Policy',
    'Philanthropic Fundraising Services': 'Non-profit, Gov & Policy',
    'Government Relations Services': 'Non-profit, Gov & Policy',
    'Professional Organizations': 'Non-profit, Gov & Policy',
    'Armed Forces': 'Non-profit, Gov & Policy',
    'Executive Offices': 'Non-profit, Gov & Policy',

    # MANUFACTURING & INFRASTRUCTURE
    'Chemical Manufacturing': 'Manufacturing & Infrastructure',
    'Chemical Raw Materials Manufacturing': 'Manufacturing & Infrastructure',
    'Agricultural Chemical Manufacturing': 'Manufacturing & Infrastructure',
    'Renewable Energy Equipment Manufacturing': 'Manufacturing & Infrastructure',
    'Engineering Services': 'Manufacturing & Infrastructure',
    'Industrial Machinery Manufacturing': 'Manufacturing & Infrastructure',
    'Manufacturing': 'Manufacturing & Infrastructure',
    'Architecture and Planning': 'Manufacturing & Infrastructure',
    'Construction': 'Manufacturing & Infrastructure',
    'Oil and Gas': 'Manufacturing & Infrastructure',
    'Utilities': 'Manufacturing & Infrastructure',
    'Automation Machinery Manufacturing': 'Manufacturing & Infrastructure',
    'Measuring and Control Instrument Manufacturing': 'Manufacturing & Infrastructure',
    'Motor Vehicle Manufacturing': 'Manufacturing & Infrastructure',
    'Food and Beverage Manufacturing': 'Manufacturing & Infrastructure',
    'Appliances, Electrical, and Electronics Manufacturing': 'Manufacturing & Infrastructure',
    'Textile Manufacturing': 'Manufacturing & Infrastructure'
}

# Apply the mapping
df_industry['Sector'] = df_industry['Industry'].map(industry_to_sector)

# Categorize anything missing as 'Other'
df_industry['Sector'] = df_industry['Sector'].fillna('Other')

# Group and sum
sector_counts = df_industry.groupby('Sector')['Total followers'].sum().sort_values(ascending=False)

# --- NUCLEATE BRANDING THEME ---
nucleate_colors = [
    '#0F054C', # Dark Navy
    '#3B298B', # Deep Purple
    '#8E85D6', # Mid Purple
    '#9DBDD8', # Light Blue
    '#C4C7F9', # Lavender
    '#ACF9E5', # Pale Teal
    '#62D0AC', # Mint
    '#3E8A81'  # Dark Teal
]

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)

# Create Donut
wedges, texts, autotexts = ax.pie(
    sector_counts.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=nucleate_colors[:len(sector_counts)],
    pctdistance=0.85, 
    wedgeprops={'width': 0.4, 'edgecolor': 'none'}
)

plt.setp(autotexts, size=12, weight="bold", color="white")

# Center Text
total_followers = sector_counts.sum()
ax.text(0, 0, f'{total_followers:,}\nTOTAL\nFOLLOWERS', 
        ha='center', va='center', 
        fontsize=20, fontweight='bold', 
        color='#000000')

# Legend
ax.legend(
    wedges, 
    sector_counts.index,
    title="Sectors",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12,
    frameon=False
)

ax.axis('equal')  
plt.title('Follower Distribution by Sector', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout()

output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig6_follower_distribution_by_sector_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"Revised sector distribution saved to {output_path}")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Load data
df_company_size = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_followers_past 365 days.xls',
    sheet_name='Company size'
)

# Define size groupings
def group_company_size(size):
    if size in ['1', '2-10']:
        return '1-10'
    elif size in ['11-50', '51-200']:
        return '11-200'
    elif size in ['201-500', '501-1000']:
        return '201-1000'
    elif size in ['1001-5000', '5001-10000']:
        return '1001-10000'
    elif size == '10001+':
        return '10001+'
    else:
        return size

# Apply grouping
df_company_size['Size Group'] = df_company_size['Company size'].apply(group_company_size)

# Group by size and sum total followers
size_counts = df_company_size.groupby('Size Group')['Total followers'].sum()

# Define logical order for grouped sizes
group_order = ['1-10', '11-200', '201-1000', '1001-10000', '10001+']
size_counts = size_counts.reindex(group_order)

# Calculate total for the center label
total_views = size_counts.sum()

# --- NUCLEATE BRANDING THEME ---
nucleate_colors = [
    '#9DBDD8', # Light Blue
    '#0F054C', # Dark Navy
    '#3B298B', # Deep Purple
    '#8E85D6', # Mid Purple
    '#C4C7F9', # Lavender
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Create the donut chart
wedges, texts, autotexts = ax.pie(
    size_counts.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=nucleate_colors,
    pctdistance=0.82, 
    wedgeprops={'width': 0.4, 'edgecolor': 'none'} # Width of the donut ring
)

# Style the percentage labels inside the rings
plt.setp(autotexts, size=15, weight="bold", color="white")

# Add the Total Followers in the center (matching the "467 Attendees" style)
ax.text(0, 0, f'{total_views:,}\nTOTAL FOLLOWERS',
        ha='center', va='center', 
        fontsize=22, fontweight='bold', 
        color='#000000', family='sans-serif')

# Add Legend to the right
ax.legend(
    wedges, 
    size_counts.index,
    title="Company Size (Employees)",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=14,
    frameon=False
)

# Title and formatting
plt.title('Follower Distribution by Company Size', 
          fontsize=26, 
          color='#000000', 
          pad=30,
          fontweight='bold')

ax.axis('equal')
plt.tight_layout()

# Save the file
output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig7_follower_distribution_by_company_size_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"\nDonut chart with company size distribution saved to: {output_path}")

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_job_function = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_followers_past 365 days.xls',
    sheet_name='Job function'
)

# TODO: revise accordingly --- SMART MAPPING DICTIONARY ---
job_map = {
    # Request: Investors, VC, & Consulting
    'Consulting': 'Investors, VC, & Consulting',
    'Operations': 'Investors, VC, & Consulting',
    'Entrepreneurship': 'Investors, VC, & Consulting',
    'Finance': 'Investors, VC, & Consulting',
    'Administrative': 'Investors, VC, & Consulting',
    'Accounting': 'Investors, VC, & Consulting',
    'Purchasing': 'Investors, VC, & Consulting',
    'Real Estate': 'Investors, VC, & Consulting',
    
    # Request: Marketing, Media, & Outreach
    'Sales': 'Marketing, Media, & Outreach',
    'Marketing': 'Marketing, Media, & Outreach',
    'Arts and Design': 'Marketing, Media, & Outreach',
    'Media and Communication': 'Marketing, Media, & Outreach',
    'Community and Social Services': 'Marketing, Media, & Outreach',
    'Customer Success and Support': 'Marketing, Media, & Outreach',
    
    # Smart Grouping: Product, Tech & Quality
    'Engineering': 'Product & Engineering',
    'Information Technology': 'Product & Engineering',
    'Quality Assurance': 'Product & Engineering',
    'Product Management': 'Product & Engineering',
    
    # Smart Grouping: Healthcare & Clinical
    'Healthcare Services': 'Healthcare & Clinical',
    
    # Legal & Management
    'Legal': 'Legal & Project Mgmt',
    'Program and Project Management': 'Legal & Project Mgmt',
    'Human Resources': 'Legal & Project Mgmt',
    'Military and Protective Services': 'Legal & Project Mgmt',
    
    # High-impact categories stay distinct
    'Research': 'Research',
    'Business Development': 'Business Development',
    'Education': 'Education'
}

# Apply mapping
df_job_function['Mapped Function'] = df_job_function['Job function'].map(job_map)

# Group by the new mapping and sum followers
job_counts = df_job_function.groupby('Mapped Function')['Total followers'].sum().sort_values(ascending=False)

# Calculate total for center label
total_views = job_counts.sum()

# --- NUCLEATE BRANDING THEME ---
nucleate_colors = [
    '#0F054C', # Dark Navy
    '#3B298B', # Deep Purple
    '#8E85D6', # Mid Purple
    '#9DBDD8', # Light Blue
    '#C4C7F9', # Lavender
    '#3E8A81', # Dark Teal
    '#62D0AC', # Mint
    '#ACF9E5', # Pale Teal
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)

# Create Donut Chart
wedges, texts, autotexts = ax.pie(
    job_counts.values,
    autopct='%1.1f%%',
    startangle=140, # Rotated for better aesthetic balance
    colors=nucleate_colors,
    pctdistance=0.82,
    wedgeprops={'width': 0.4, 'edgecolor': 'none'}
)

# Style internal percentage labels
plt.setp(autotexts, size=12, weight="bold", color="white")

# Center Text (Nucleate Style)
ax.text(0, 0, f'{total_views:,}\nTOTAL FOLLOWERS', 
        ha='center', va='center', 
        fontsize=20, fontweight='bold', 
        color='#000000')

# Legend on the right
ax.legend(
    wedges, 
    job_counts.index,
    title="Job Functions",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12,
    frameon=False
)

plt.title('Visitor Distribution by Job Function', 
          fontsize=26, 
          color='#000000', 
          pad=30,
          fontweight='bold')

ax.axis('equal')
plt.tight_layout()

# Save
output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig8_followers_distribution_by_job_function_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"Mapped Job Function chart saved to: {output_path}")

