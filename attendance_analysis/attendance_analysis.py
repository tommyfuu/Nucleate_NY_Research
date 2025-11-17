import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# ===== COMPLETE EVENT DATA FROM SHEET 1 =====
complete_events_data = {
    'Date': ['10/10/2024', '10/16/2024', '10/17/2024', '11/5/2024', '11/12/2024', 
             '11/18/2024', '11/20/2024', '12/5/2024', '12/10/2024', '1/15/2025',
             '1/21/2025', '1/30/2025', '2/4/2025', '2/18/2025', '2/19/2025',
             '2/24/2025', '3/4/2025', '3/5/2025', '3/18/2025', '4/1/2025',
             '4/17/2025', '4/24/2025', '4/28/2025', '5/1/2025', '5/6/2025',
             '5/8/2025', '5/16/2025', '5/16/2025', '9/10/2025', '9/25/2025'],
    'Event': ['Nucleate NY 4.0 Kickoff', 'BioHack NYC', 'Bio Climate Deep Tech Mixer',
              'Activator HH 11/5', 'Activator HH 11/12', 'Team Matching Virtual',
              'Team Matching In-Person', 'Bridging the Gap Workshop', 'NIH Funding Workshop',
              'Launching Tech Ventures', 'Team Launch', 'Mentor Matching',
              'Workshop 1: Technical Risk', 'Workshop 2: Market Risk', 'Deep Origin Mixer',
              'Creative Toolkit Workshop', 'Workshop 3: Inclusive Strategies', 
              'Bio Climate Deep Tech Salon', 'Workshop 4: Strategic Fundraising',
              'Workshop 5: Regulatory Strategy', 'Activator Seminar: Presentations',
              'Activator Alumni Showcase', 'Practice Pitch Virtual', 'BlueRock HH Mixer',
              'Practice Pitch In-Person', 'Activator Closeout', 'Forum AM', 'Forum PM',
              'Made in NYC', 'Fall Mixer with NYAS'],
    'Attendees': [210, 227, 211, 61, 92, 84, 60, 55, 59, 102,
                  81, 104, 99, 86, 211, 12, 117, 186, 85, 80,
                  121, 461, 30, 202, 36, 127, 273, 300, 365, 85]
}

df_all_events = pd.DataFrame(complete_events_data)
df_all_events['Date'] = pd.to_datetime(df_all_events['Date'])

# 0. plot the number of events per months
df_all_events['Month'] = df_all_events['Date'].dt.to_period('M')
events_per_month = df_all_events.groupby('Month').size().reset_index(name='Number of Events')
plt.figure(figsize=(16, 8))
sns.barplot(x='Month', y='Number of Events', data=events_per_month, palette='magma')
plt.title('Number of Nucleate Events per Month (2024-2025)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Events', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/events_per_month.png')

# 1. plot the attendance by numbers in barcharts from high to low from left to right; event names on x-axis
df_sorted = df_all_events.sort_values(by='Attendees', ascending=False)
plt.figure(figsize=(16, 8))
sns.barplot(x='Event', y='Attendees', data=df_sorted, palette='coolwarm')
plt.title('Attendance at Nucleate Events (2024-2025)', fontsize=16)
plt.xlabel('Event', fontsize=14)
plt.ylabel('Number of Attendees', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/attendance_by_event.png')

# 2. Plot average attendance number per event by months
df_all_events['Month'] = df_all_events['Date'].dt.to_period('M')
average_attendance = df_all_events.groupby('Month')['Attendees'].mean().reset_index()

# Convert Period to string for plotting
average_attendance['Month_str'] = average_attendance['Month'].astype(str)

plt.figure(figsize=(16, 8))
# Use matplotlib directly or convert Month to string
plt.plot(average_attendance['Month_str'], average_attendance['Attendees'], 
         marker='o', linewidth=2, markersize=10, color='teal')
plt.title('Average Attendance at Nucleate Events per Month (2024-2025)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Number of Attendees', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/average_attendance_per_month.png', 
            dpi=300, bbox_inches='tight')


# 4. plot percentage distribution of sectors as line plot - x-axis sectors, y-axis percentage, events represented by different lines in color
event_data = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/nucleate_column_data.csv', index_col=0)
## columns are sectors, indexes are events
### remove other as a sector
event_data = event_data.drop(columns=['Other'])
sectors = event_data.columns.tolist()
percentage_data = event_data.div(event_data.sum(axis=1), axis=0) * 100
plt.figure(figsize=(16, 8))
for event in percentage_data.index:
    plt.plot(sectors, percentage_data.loc[event], marker='o', label=event)
plt.title('Percentage Distribution of Sectors by Event', fontsize=16)
plt.xlabel('Sectors', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Events', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/percentage_distribution_sectors.png', 
            dpi=300, bbox_inches='tight')


# 5. plot interest form in terms of the number per category
### the three columns are Affiliation, Number, Category, the Category is , -delimited string
interest_form_data = pd.read_csv('/data/morrisq/fuc/nucleate_research/attendance_analysis/data/interest_form.csv')
# Explode the Category column to have one category per row
interest_form_exploded = interest_form_data.assign(
    Category=interest_form_data['Category'].str.split(', ')
).explode('Category')
# Group by Category and sum the Number
# sort by descending order
category_counts = interest_form_exploded.groupby('Category')['Number'].sum().reset_index()
category_counts = category_counts.sort_values(by='Number', ascending=False)
plt.figure(figsize=(16, 8))
sns.barplot(x='Category', y='Number', data=category_counts, palette='viridis')
plt.title('Interest Form Responses by Category', fontsize=16)
plt.xlabel('Source Category', fontsize=14)
plt.ylabel('Number of Responses', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/data/morrisq/fuc/nucleate_research/attendance_analysis/vis/interest_form_by_category.png', 
            dpi=300, bbox_inches='tight')

