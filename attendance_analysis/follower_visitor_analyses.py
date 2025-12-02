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

# Create a geocoding function with error handling
def geocode_location(location_name, geolocator, max_retries=3):
    """Geocode a location with retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Rate limiting
            location = geolocator.geocode(location_name, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except GeocoderTimedOut:
            if attempt == max_retries - 1:
                return None, None
            time.sleep(2)
    return None, None

# Initialize geocoder
geolocator = Nominatim(user_agent="nucleate_location_analysis")

# Add coordinates to dataframe
df_location['latitude'] = None
df_location['longitude'] = None

# print("\nGeocoding locations...")
# for idx, row in df_location.iterrows():
#     location_name = row['Location']
#     print(f"Processing: {location_name}")
#     lat, lon = geocode_location(location_name, geolocator)
#     df_location.at[idx, 'latitude'] = lat
#     df_location.at[idx, 'longitude'] = lon

# # Remove rows without coordinates
# df_map = df_location.dropna(subset=['latitude', 'longitude'])

# # save the geocoded data for future use
# df_map.to_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate_ny_visitor_locations_geocoded_past_365_days.csv', index=False)

df_map = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate_ny_visitor_locations_geocoded_past_365_days.csv')
print(f"\nSuccessfully geocoded: {len(df_map)}/{len(df_location)} locations")
print(df_map[['Location', 'Total views', 'latitude', 'longitude']])

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
    ),
    height=600,
    title_font_size=16
)
fig1.write_image('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/nucleate_ny_visitor_locations_bubble_map.png', scale=2)

### analysis 3. Industry tab
import pandas as pd
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

# Apply the mapping to your dataframe
# Assuming your dataframe has an 'Industry' column
df_industry['Sector'] = df_industry['Industry'].map(industry_to_sector)

# Check for any unmapped industries
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
sector_counts = df_industry['Sector'].value_counts()
print(sector_counts)

# Create a visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
sector_counts.plot(kind='bar', color='skyblue')
plt.title('Visitor Distribution by Sector', fontsize=16)
plt.xlabel('Sector', fontsize=14)
plt.ylabel('Number of Visitors', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/visitor_distribution_by_sector.png', dpi=300, bbox_inches='tight')

### analysis 4. tab Company Size
# Company size	Total views
# 1	89
# 2-10	955
# 10001+	2199
# 501-1000	402
# 11-50	1130
# 51-200	794
# 5001-10000	505
# 1001-5000	1037
# 201-500	490

df_company_size = pd.read_excel(
    '/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate-ny_visitors_past and this cycle past 365 days.xls',
    sheet_name='Company size'
)
# Sort company sizes in a logical order
size_order = ['1', '2-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10001+']
df_company_size['Company size'] = pd.Categorical(df_company_size['Company size'], categories=size_order, ordered=True)
df_company_size = df_company_size.sort_values('Company size')
plt.figure(figsize=(12, 8))
plt.bar(df_company_size['Company size'], df_company_size['Total views'], color='lightgreen')
plt.title('Visitor Distribution by Company Size', fontsize=16)
plt.xlabel('Company Size', fontsize=14)
plt.ylabel('Total Views', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/followers_visitors/visitor_distribution_by_company_size.png', dpi=300, bbox_inches='tight')