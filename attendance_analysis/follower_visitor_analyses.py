import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

## read the Visitor metrics tab
df_followers_visitors = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
    sheet_name='Visitor metrics'
)
### analysis 1. plot the month-by-month 'Overview page views (total)' and 'Overview unique visitors (total)' as two lines in one plot
### has a date column 'Date' in format 11/16/2024'
df_followers_visitors['Month'] = pd.to_datetime(df_followers_visitors['Date']).dt.to_period('M').astype(str)
## sum up by month
df_followers_visitors = df_followers_visitors.groupby('Month').agg({
    'Overview page views (total)': 'sum',
    'Overview unique visitors (total)': 'sum'
}).reset_index()
plt.figure(figsize=(16, 8))
plt.plot(df_followers_visitors['Month'], df_followers_visitors['Overview page views (total)'],
         marker='o', label='Overview Page Views (Total)', color='blue')
plt.plot(df_followers_visitors['Month'], df_followers_visitors['Overview unique visitors (total)'],
            marker='o', label='Overview Unique Visitors (Total)', color='orange')
plt.title('Monthly Overview Page Views and Unique Visitors', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/monthly_overview_page_views_unique_visitors.png', dpi=300, bbox_inches='tight')


### analysis 2, given the Location tab, visualize on a map those numbers
# Location	Total views
# Washington DC-Baltimore Area
# Atlanta Metropolitan Area
# Athens Metropolitan Area, Greece
# Miami-Fort Lauderdale Area
# Greater Chicago Area
# Galway Metropolitan Area, Ireland
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
## using conda env test_env_dup

# Read the data
df_location = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
    sheet_name='Location'
)

# Display the data structure
print("Data preview:")
print(df_location.head(10))
print(f"\nShape: {df_location.shape}")
print(f"\nColumns: {df_location.columns.tolist()}")

# Initialize geocoder (Keep this outside the function)
geolocator = Nominatim(user_agent="nucleate_location_analysis")

def simplify_location_name(location_name):
    """
    Attempts to simplify complex metropolitan area names to a primary city.

    Examples:
    - "Washington DC-Baltimore Area" -> "Washington DC"
    - "Miami-Fort Lauderdale Area" -> "Miami"
    - "Greater Chicago Area" -> "Chicago"
    - "Athens Metropolitan Area, Greece" -> "Athens, Greece"
    """
    # 1. Handle common suffixes and separators
    # Remove " Area", "Greater ", " Metropolitan"
    simplified = re.sub(r'(\sArea|\sGreater\s|\sMetropolitan)', '', location_name).strip()
    
    # Take the first city in a hyphenated/slashed list (e.g., "Washington DC-Baltimore")
    simplified = simplified.split('-')[0].split('/')[0].strip()

    # 2. Add country/region context for better accuracy if missing, 
    # but only for known international cases like "Athens"
    if simplified == "Athens" and "Greece" not in location_name:
        return "Athens, Greece"
        
    return simplified

def geocode_location(location_name, geolocator, max_retries=3):
    """
    Geocode a location with retry logic and a simplification fallback.
    """
    # List of names to try, in order of preference
    location_attempts = [location_name]
    
    # Create the simplified name and add it as a fallback if it's different
    simplified_name = simplify_location_name(location_name)
    if simplified_name != location_name:
        location_attempts.append(simplified_name)
    
    # Add the specific Boston handling to the most relevant attempt
    if "Boston" in location_name and "Boston, MA, USA" not in location_attempts:
         # Prioritize the most specific accurate attempt
        location_attempts.insert(0, "Boston, MA, USA") 

    for name_to_try in location_attempts:
        print(f"  Attempting to geocode: **{name_to_try}**")
        
        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Rate limiting
                location = geolocator.geocode(name_to_try, timeout=10)
                
                if location:
                    print(f"  ✅ SUCCESS for {location_name} using: {name_to_try}")
                    return location.latitude, location.longitude
                else:
                    # If geocode returns None, break the retry loop for this specific name_to_try 
                    # and move to the next fallback name (if any)
                    break 
            except GeocoderTimedOut:
                print(f"  ⚠️ Geocoding timed out for {name_to_try}, retrying... (Attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    print(f"  ❌ Failed after max retries for {name_to_try}.")
                    break # Break retry loop, move to next fallback name
                time.sleep(2)
            except Exception as e:
                # Catch other potential Geopy/network errors
                print(f"  An error occurred for {name_to_try}: {e}")
                break # Break retry loop, move to next fallback name
    
    # If all attempts and fallbacks fail
    print(f"  ❌ FAILED to find coordinates for: {location_name}")
    return None, None

# Add coordinates to dataframe
df_location['latitude'] = None
df_location['longitude'] = None

print("\nGeocoding locations...")
for idx, row in df_location.iterrows():
    location_name = row['Location']
    print(f"\nProcessing Original Location: {location_name}")
    lat, lon = geocode_location(location_name, geolocator)
    df_location.at[idx, 'latitude'] = lat
    df_location.at[idx, 'longitude'] = lon

# # Remove rows without coordinates
# df_map = df_location.dropna(subset=['latitude', 'longitude'])

# for the ones without coordinates, try to geocode again with modified names
import re

def simplify_location_name(location_name):
    """
    Attempts to simplify complex metropolitan area names to a primary city.
    """
    # 1. Handle "Boston" edge case first (though it's handled in geocode_location too)
    if "Boston" in location_name:
        return "Boston, MA, USA"

    # 2. Remove common descriptive terms aggressively
    simplified = re.sub(
        r'(\sThe|\sArea|\sGreater\s|\sMetropolitan|\sRegion|\sMetroplex|\sBay)', 
        '', 
        location_name, 
        flags=re.IGNORECASE
    ).strip()
    
    # 3. Handle duplicate country/region names caused by sheet data
    # Example: "London, United Kingdom, United Kingdom" -> "London, United Kingdom"
    parts = [p.strip() for p in simplified.split(',')]
    if len(parts) >= 3 and parts[-1] == parts[-2]:
        simplified = ', '.join(parts[:-1])
        
    # 4. Take the first city in a hyphenated/slashed list (e.g., "New Bern-Morehead City")
    simplified = simplified.split('-')[0].split('/')[0].strip()

    # 5. Specific country/region context corrections
    if simplified.lower() == "athens" and "greece" not in location_name.lower():
        return "Athens, Greece"
    elif 'randstad' in simplified.lower():
        return "Amsterdam, Netherlands"
    
    # Example: Greater Terni, Italy -> Terni, Italy
    if simplified.startswith("Greater "):
        simplified = simplified.replace("Greater ", "")

    return simplified

df_map = df_location.copy()
for idx, row in df_map[df_map['latitude'].isna() | df_map['longitude'].isna()].iterrows():
    location_name = row['Location']
    print(f"Re-processing: {location_name}")
    # TODO: revise location_name to make it work
    lat, lon = geocode_location(simplify_location_name(location_name), geolocator, max_retries=5)
    df_map.at[idx, 'latitude'] = lat
    df_map.at[idx, 'longitude'] = lon

# # save the geocoded data for future use
# df_map.to_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate_ny_visitor_locations_geocoded_past_365_days.csv', index=False)

df_map = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate_ny_visitor_locations_geocoded_past_365_days.csv')
print(f"\nSuccessfully geocoded: {len(df_map)}/{len(df_location)} locations")
print(df_map[['Location', 'Total views', 'latitude', 'longitude']])

# sort by 'Total views' descending
df_map = df_map.sort_values(by='Total views', ascending=False)

# visualize all with over 100 views
df_map = df_map[df_map['Total views'] >= 100]


# Visualization 1: Interactive Bubble Map with Plotly
fig1 = px.scatter_geo(
    df_map,
    lat='latitude',
    lon='longitude',
    size='Total views',
    hover_name='Location',
    hover_data={'Total views': True, 'latitude': False, 'longitude': False},
    title='Nucleate NY Visitor Locations (Past 365 Days)',
    size_max=50,
    projection='natural earth'
)

fig1.update_layout(
    geo=dict(
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        projection_scale=1.5,
        showframe=False, 
    ),
    height=600,
    title_font_size=16
)
# save to html
fig1.write_html('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/nucleate_ny_visitor_locations_bubble_map.html')

fig1.write_image('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/nucleate_ny_visitor_locations_bubble_map.png', scale=2)
### analysis 3. Industry tab - PIE CHART VERSION
import pandas as pd
import matplotlib.pyplot as plt

df_industry = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
    sheet_name='Industry'
)

# Define sector mapping dictionary
industry_to_sector = {
    # HEALTHCARE & LIFE SCIENCES
    'Health and Human Services': 'Healthcare & Life Sciences',
    'Medical and Diagnostic Laboratories': 'Healthcare & Life Sciences',
    'Biotechnology Research': 'Healthcare & Life Sciences',
    'Medical Practices': 'Healthcare & Life Sciences',
    'Hospitals and Health Care': 'Healthcare & Life Sciences',
    'Pharmaceutical Manufacturing': 'Healthcare & Life Sciences',
    'Medical Equipment Manufacturing': 'Healthcare & Life Sciences',
    'Retail Pharmacies': 'Healthcare & Life Sciences',
    'Public Health': 'Healthcare & Life Sciences',
    'Wellness and Fitness Services': 'Healthcare & Life Sciences',
    'Retail Health and Personal Care Products': 'Healthcare & Life Sciences',
    
    # TECHNOLOGY & IT
    'Online Audio and Video Media': 'Technology & IT',
    'Nanotechnology Research': 'Technology & IT',
    'Computer and Network Security': 'Technology & IT',
    'IT Services and IT Consulting': 'Technology & IT',
    'Technology, Information and Media': 'Technology & IT',
    'Climate Data and Analytics': 'Technology & IT',
    'Robotics Engineering': 'Technology & IT',
    'Software Development': 'Technology & IT',
    'Technology, Information and Internet': 'Technology & IT',
    'Telecommunications': 'Technology & IT',
    'Data Infrastructure and Analytics': 'Technology & IT',
    
    # FINANCIAL & PROFESSIONAL SERVICES
    'Real Estate': 'Financial & Professional Services',
    'Investment Banking': 'Financial & Professional Services',
    'Investment Management': 'Financial & Professional Services',
    'Accounting': 'Financial & Professional Services',
    'Market Research': 'Financial & Professional Services',
    'Public Relations and Communications Services': 'Financial & Professional Services',
    'Legal Services': 'Financial & Professional Services',
    'Business Consulting and Services': 'Financial & Professional Services',
    'Operations Consulting': 'Financial & Professional Services',
    'Human Resources Services': 'Financial & Professional Services',
    'Administrative and Support Services': 'Financial & Professional Services',
    'Strategic Management Services': 'Financial & Professional Services',
    'Government Relations Services': 'Financial & Professional Services',
    'Staffing and Recruiting': 'Financial & Professional Services',
    'Venture Capital and Private Equity Principals': 'Financial & Professional Services',
    'Advertising Services': 'Financial & Professional Services',
    'Insurance': 'Financial & Professional Services',
    'Financial Services': 'Financial & Professional Services',
    'Law Practice': 'Financial & Professional Services',
    
    # MANUFACTURING & ENGINEERING
    'Renewable Energy Equipment Manufacturing': 'Manufacturing & Engineering',
    'Engineering Services': 'Manufacturing & Engineering',
    'Chemical Manufacturing': 'Manufacturing & Engineering',
    'Defense and Space Manufacturing': 'Manufacturing & Engineering',
    'Industrial Machinery Manufacturing': 'Manufacturing & Engineering',
    'Manufacturing': 'Manufacturing & Engineering',
    'Architecture and Planning': 'Manufacturing & Engineering',
    'Civil Engineering': 'Manufacturing & Engineering',
    'Design Services': 'Manufacturing & Engineering',
    'Food and Beverage Manufacturing': 'Manufacturing & Engineering',
    
    # CONSUMER & RETAIL
    'Consumer Services': 'Consumer & Retail',
    'Retail': 'Consumer & Retail',
    'Entertainment Providers': 'Consumer & Retail',
    'Food and Beverage Services': 'Consumer & Retail',
    'Transportation, Logistics, Supply Chain and Storage': 'Consumer & Retail',
    'Events Services': 'Consumer & Retail',
    'Marketing Services': 'Consumer & Retail',
    'Business Content': 'Consumer & Retail',
    
    # EDUCATION, GOVERNMENT & NON-PROFIT
    'Primary and Secondary Education': 'Education, Government & Non-Profit',
    'Higher Education': 'Education, Government & Non-Profit',
    'Education Administration Programs': 'Education, Government & Non-Profit',
    'Education': 'Education, Government & Non-Profit',
    'Philanthropic Fundraising Services': 'Education, Government & Non-Profit',
    'Government Administration': 'Education, Government & Non-Profit',
    'Artists and Writers': 'Education, Government & Non-Profit',
    'Book Publishing': 'Education, Government & Non-Profit',
    'Non-profit Organizations': 'Education, Government & Non-Profit',
    'Writing and Editing': 'Education, Government & Non-Profit',
    'Libraries': 'Education, Government & Non-Profit',
    'Environmental Services': 'Education, Government & Non-Profit',
    'Research Services': 'Education, Government & Non-Profit',
}

# Apply the mapping
df_industry['Sector'] = df_industry['Industry'].map(industry_to_sector)

# Check for unmapped industries
unmapped = df_industry[df_industry['Sector'].isna()]['Industry'].unique()
if len(unmapped) > 0:
    print("Unmapped industries found:")
    print(unmapped)
else:
    print("All industries successfully mapped!")

# Display sector distribution
print("\n" + "="*50)
print("SECTOR DISTRIBUTION")
print("="*50)
# sector_counts = df_industry['Sector'].value_counts()
# sector counts should be sum of 'Total views' per sector
sector_counts = df_industry.groupby('Sector')['Total views'].sum().sort_values(ascending=False)
print(sector_counts)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- NUCLEATE BRANDING THEME ---
# Colors extracted from the provided image
nucleate_colors = [
    '#9DBDD8', # Light Blue (New to Nucleate)
    '#0F054C', # Dark Navy (Alumni)
    '#3B298B', # Deep Purple (Sponsor)
    '#8E85D6', # Mid Purple (Activator)
    '#C4C7F9', # Lavender (Leadership)
    '#ACF9E5', # Pale Teal (Expert/Mentor)
    '#62D0AC', # Mint (Other)
    '#3E8A81'  # Dark Teal (Speaker)
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)

# Create the donut chart
# 'pctdistance' moves the percentage labels inside the ring
wedges, texts, autotexts = ax.pie(
    sector_counts.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=nucleate_colors[:len(sector_counts)],
    pctdistance=0.85, 
    wedgeprops={'width': 0.4, 'edgecolor': 'none'} # 'width' creates the donut hole
)

# Style the percentage labels (the white numbers inside the slices)
plt.setp(autotexts, size=14, weight="bold", color="white")

# Add the Legend to the right (matching the image layout)
ax.legend(
    wedges, 
    sector_counts.index,
    title="Sectors",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=14,
    frameon=False
)

# Set equal aspect ratio to ensure it's a circle
ax.axis('equal')  

plt.title('Visitor Distribution by Sector', 
          fontsize=24, 
          color='#000000', 
          pad=20,
          fontweight='bold')

plt.tight_layout()

# Save the file
output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig1_visitor_distribution_by_sector_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"\nDonut chart styled like Nucleate branding saved to: {output_path}")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Load data
df_company_size = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
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

# Group by size and sum total views
size_counts = df_company_size.groupby('Size Group')['Total views'].sum()

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

# Add the Total Views in the center (matching the "467 Attendees" style)
ax.text(0, 0, f'{total_views:,}\nTOTAL VIEWS', 
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
plt.title('Visitor Distribution by Company Size', 
          fontsize=26, 
          color='#000000', 
          pad=30,
          fontweight='bold')

ax.axis('equal')
plt.tight_layout()

# Save the file
output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig2_visitor_distribution_by_company_size_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"\nDonut chart with company size distribution saved to: {output_path}")

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_job_function = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
    sheet_name='Job function'
)

# --- SMART MAPPING DICTIONARY ---
job_map = {
    # Request: Investors, VC, & Consulting
    'Consulting': 'Investors, VC, & Consulting',
    'Operations': 'Investors, VC, & Consulting',
    'Entrepreneurship': 'Investors, VC, & Consulting',
    'Finance': 'Investors, VC, & Consulting',
    'Administrative': 'Investors, VC, & Consulting',
    
    # Request: Marketing, Media, & Outreach
    'Sales': 'Marketing, Media, & Outreach',
    'Marketing': 'Marketing, Media, & Outreach',
    'Arts and Design': 'Marketing, Media, & Outreach',
    'Media and Communication': 'Marketing, Media, & Outreach',
    'Community and Social Services': 'Marketing, Media, & Outreach',
    
    # Smart Grouping: Product & Tech
    'Engineering': 'Product & Engineering',
    'Information Technology': 'Product & Engineering',
    'Quality Assurance': 'Product & Engineering',
    'Product Management': 'Product & Engineering',
    'Healthcare Services': 'Product & Engineering',  # Some healthcare roles are very product/tech-focused (e.g., digital health)
    
    # Legal & Management
    'Legal': 'Legal & Project Mgmt',
    'Program and Project Management': 'Legal & Project Mgmt',
    'Human Resources': 'Legal & Project Mgmt',
    
    # High-impact categories stay distinct
    'Research': 'Research',
    'Business Development': 'Business Development',
    'Education': 'Education'
}

# Apply mapping
df_job_function['Mapped Function'] = df_job_function['Job function'].map(job_map)

# Group by the new mapping and sum views
job_counts = df_job_function.groupby('Mapped Function')['Total views'].sum().sort_values(ascending=False)

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
ax.text(0, 0, f'{total_views:,}\nTOTAL VIEWS', 
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
output_path = '/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/final_figs/fig3_visitor_distribution_by_job_function_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"Mapped Job Function chart saved to: {output_path}")



### Event Attendance Analysis - PIE CHART VERSION
import pandas as pd
import matplotlib.pyplot as plt

# Load event data
event_data = pd.read_csv('./data/nucleate_column_data.csv', index_col=0)

# 1. AGGREGATE AND CONSOLIDATE CATEGORIES
category_totals = event_data.sum(axis=0)

# Define the consolidation mapping
# We combine HealthTech, Software, Tools & Reagents, and CRO-CDMO into Biotech / Pharma
consolidation_map = {
    'HealthTech / Software': 'Biotech / Pharma',
    'Tools & Reagents / CRO-CDMO': 'Biotech / Pharma',
}

# Group the series by the map (keep original name if not in map)
category_totals = category_totals.groupby(lambda x: consolidation_map.get(x, x)).sum()
category_totals = category_totals.sort_values(ascending=False)

# Calculate grand total for center label
grand_total = category_totals.sum()

# --- 2. NUCLEATE BRANDING THEME ---
nucleate_colors = [
    '#0F054C', # Dark Navy
    '#3B298B', # Deep Purple
    '#8E85D6', # Mid Purple
    '#9DBDD8', # Light Blue
    '#C4C7F9', # Lavender
    '#3E8A81', # Dark Teal
    '#62D0AC', # Mint
    '#ACF9E5'  # Pale Teal
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# --- 3. CREATE DONUT CHART ---
wedges, texts, autotexts = ax.pie(
    category_totals.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=nucleate_colors[:len(category_totals)],
    pctdistance=0.82, 
    wedgeprops={'width': 0.4, 'edgecolor': 'none'}
)

# Style internal percentage labels
plt.setp(autotexts, size=14, weight="bold", color="white")

# Center Text (Nucleate Style)
ax.text(0, 0, f'{grand_total:,}\nTOTAL\nATTENDEES', 
        ha='center', va='center', 
        fontsize=22, fontweight='bold', 
        color='#000000', family='sans-serif')

# Legend on the right
ax.legend(
    wedges, 
    category_totals.index,
    title="Attendee Categories",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=13,
    frameon=False
)

# Add title
plt.title('Aggregated Event Attendance by Category', 
          fontsize=26, 
          color='#000000', 
          pad=30,
          fontweight='bold')

ax.axis('equal')
plt.tight_layout()

# Save output
output_path = './vis/final_figs/fig4_aggregated_event_attendance_consolidated_donut.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
plt.show()

print(f"\nConsolidated Donut chart saved to: {output_path}")
print("Categories merged into Biotech / Pharma: HealthTech, Software, Tools & Reagents, CRO-CDMO")

import pandas as pd
import matplotlib.pyplot as plt

# Load partnership data
partnership_data = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/interest_form.csv')

# Initialize a dictionary to store category counts
category_counts = {}

# Process each row, splitting categories that are comma-separated
for _, row in partnership_data.iterrows():
    # Handle potential non-string values and split
    raw_cat = str(row['Category'])
    categories = [cat.strip() for cat in raw_cat.split(',')]
    number = row['Number']
    
    # Distribute the count across all categories
    for category in categories:
        if category in category_counts:
            category_counts[category] += number
        else:
            category_counts[category] = number

# Convert to Series and sort
category_totals = pd.Series(category_counts).sort_values(ascending=False)

# Group smaller categories into "Other" (Threshold = 15)
threshold = 15
category_filtered = category_totals[category_totals >= threshold].copy()
other_sum = category_totals[category_totals < threshold].sum()
if other_sum > 0:
    category_filtered['Other'] = other_sum
category_totals = category_filtered

# Calculate total for center label
grand_total = category_totals.sum()

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
    '#7C9FA8', # Slate Gray (Secondary)
    '#B8D4DC'  # Pale Blue (Secondary)
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Create the Donut Chart
wedges, texts, autotexts = ax.pie(
    category_totals.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=nucleate_colors[:len(category_totals)],
    pctdistance=0.82, 
    wedgeprops={'width': 0.4, 'edgecolor': 'none'}
)

# Style internal percentage labels
plt.setp(autotexts, size=13, weight="bold", color="white")

# Add the Central Total (Nucleate Style)
ax.text(0, 0, f'{int(grand_total):,}\nTOTAL\nPARTNERS', 
        ha='center', va='center', 
        fontsize=22, fontweight='bold', 
        color='#000000', family='sans-serif')

# Add Legend to the right
ax.legend(
    wedges, 
    category_totals.index,
    title="Partner Categories",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12,
    frameon=False
)

# Set title
plt.title('Partnership Distribution by Category', 
          fontsize=28, 
          color='#000000', 
          pad=30,
          fontweight='bold')

ax.axis('equal')
plt.tight_layout()

# Save output
output_path = './vis/final_figs/fig5_partnership_distribution_donut_branded.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
plt.show()

print(f"\nBranded Donut chart saved to: {output_path}")