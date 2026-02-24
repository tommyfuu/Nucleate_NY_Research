import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set PDF/PS settings for high-quality export (Vector Graphics)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- 1. GLOBAL BRANDING & COLOR CONFIGURATION ---
# Fixed mapping ensures specific categories always use the same color
category_colors = {
    'Academia': '#0F054C',                      # Dark Navy
    'Biotech/Healthtech/Pharma': '#3B298B',      # Deep Purple
    'Investors, VC, & Consulting': '#8E85D6',    # Mid Purple
    'Startups': '#3E8A81',                       # Dark Teal
    'Product & Engineering': '#9DBDD8',          # Light Blue
    'Marketing & Outreach': '#C4C7F9',           # Lavender
    'Legal & Project Mgmt': '#62D0AC',           # Mint
    'Education': '#ACF9E5',                      # Pale Teal
    'Other': '#D3D3D3'                           # Light Grey
}

def create_donut_chart(data, total_label, title, output_path):
    """Standardized function to create the Nucleate-styled donut chart."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_alpha(0.0)
    
    # Map colors from the global dictionary based on the index labels
    current_colors = [category_colors.get(cat, '#CCCCCC') for cat in data.index]
    
    wedges, texts, autotexts = ax.pie(
        data.values,
        autopct='%1.1f%%',
        startangle=140,
        colors=current_colors,
        pctdistance=0.82,
        wedgeprops={'width': 0.4, 'edgecolor': 'none'}
    )
    
    plt.setp(autotexts, size=12, weight="bold", color="white")
    
    # Center Text (Grand Total)
    ax.text(0, 0, f'{int(data.sum()):,}\n{total_label}', 
            ha='center', va='center', 
            fontsize=20, fontweight='bold', color='#000000')

    # Legend Configuration
    ax.legend(wedges, data.index, title="Categories", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, frameon=False)

    plt.title(title, fontsize=26, color='#000000', pad=30, fontweight='bold')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Successfully saved: {output_path}")

# --- 2. COMMON MAPPING LOGIC ---
# Standardizes raw LinkedIn/Input labels to your requested Master Categories
master_map = {
    # Research & Education
    'Research': 'Academia',
    'Education': 'Academia',
    'University': 'Academia',
    'Academia': 'Academia',
    
    # Biotech/Healthtech/Pharma
    'Business Development': 'Biotech/Healthtech/Pharma',
    'Biotech / Pharma': 'Biotech/Healthtech/Pharma',
    'Biotech/pharma companies': 'Biotech/Healthtech/Pharma',
    'HealthTech / Software': 'Biotech/Healthtech/Pharma',
    'Tools & Reagents / CRO-CDMO': 'Biotech/Healthtech/Pharma',
    'Healthcare Services': 'Biotech/Healthtech/Pharma',
    'Healthcare Provider': 'Biotech/Healthtech/Pharma',
    
    # Investors, VC, & Consulting
    'Consulting': 'Investors, VC, & Consulting',
    'Consulting firm': 'Investors, VC, & Consulting',
    'Finance': 'Investors, VC, & Consulting',
    'VC/equity/incubator': 'Investors, VC, & Consulting',
    'VC / Investor': 'Investors, VC, & Consulting',
    'Venture Capital': 'Investors, VC, & Consulting',
    'Provider': 'Investors, VC, & Consulting',
    'Entrepreneurship': 'Investors, VC, & Consulting',
    'Operations': 'Investors, VC, & Consulting',
    'Administrative': 'Investors, VC, & Consulting',
    'Accounting': 'Investors, VC, & Consulting',
    
    # Startups
    'Early-to-growth startup': 'Startups',
    
    # Product & Engineering
    'Engineering': 'Product & Engineering',
    'Information Technology': 'Product & Engineering',
    'Quality Assurance': 'Product & Engineering',
    'Product Management': 'Product & Engineering',
    
    # Marketing & Outreach
    'Sales': 'Marketing & Outreach',
    'Marketing': 'Marketing & Outreach',
    'Media and Communication': 'Marketing & Outreach',
    'Arts and Design': 'Marketing & Outreach',
    'Non-profit': 'Marketing & Outreach',
    
    # Legal & Regulatory & Management
    'Legal': 'Legal & Project Mgmt',
    'Program and Project Management': 'Legal & Project Mgmt',
    'Human Resources': 'Legal & Project Mgmt',
    'Law firm': 'Legal & Project Mgmt',
    'Government': 'Legal & Project Mgmt',
}

# --- 3. EXECUTION: FIGURE 1 - VISITORS ---
df_vis = pd.read_excel('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls', sheet_name='Job function')
df_vis['Mapped'] = df_vis['Job function'].map(master_map).fillna('Other')
vis_counts = df_vis.groupby('Mapped')['Total views'].sum().sort_values(ascending=False)
create_donut_chart(vis_counts, "TOTAL\nVIEWS", "Visitor Distribution by Job Function", './vis/final_figs/FINAL_fig1_visitor_distribution.pdf')

# --- 4. EXECUTION: FIGURE 2 - FOLLOWERS ---
df_fol = pd.read_excel('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_followers_past 365 days.xls', sheet_name='Job function')
df_fol['Mapped'] = df_fol['Job function'].map(master_map).fillna('Other')
fol_counts = df_fol.groupby('Mapped')['Total followers'].sum().sort_values(ascending=False)
create_donut_chart(fol_counts, "TOTAL\nFOLLOWERS", "Follower Distribution by Job Function", './vis/final_figs/FINAL_fig2_follower_distribution.pdf')

# --- 5. EXECUTION: FIGURE 3 - EVENT ---
df_part = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/interest_form.csv')
part_counts_raw = {}
for _, row in df_part.iterrows():
    cats = [c.strip() for c in str(row['Category']).split(',')]
    for c in cats:
        part_counts_raw[c] = part_counts_raw.get(c, 0) + row['Number']

part_series = pd.Series(part_counts_raw)
part_final = part_series.groupby(lambda x: master_map.get(x, 'Other')).sum().sort_values(ascending=False)
create_donut_chart(part_final, "TOTAL\nATTENDANTS", "Event Attendant Distribution by Category", './vis/final_figs/FINAL_fig3_event_distribution.pdf')