
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\DataSetAnalysis\shark data\attacks.csv", encoding='latin-1')
# Display first few rows to verify columns and data
print(df.head())

def assign_daypart(time_str):
    """
    Clean the "Time" column and categorize times into broader day parts.
    Expected dayparts: Morning, Afternoon, Evening, Night.
    """
    if pd.isna(time_str):
        return np.nan

    # Try to extract the hour from the time string
    try:
        # Remove non-numeric parts (e.g., 'h' or extra words) and extract the hour
        hour = re.findall(r'\d+', time_str)
        if hour:
            hour = int(hour[0])
        else:
            return np.nan
    except Exception:
        return np.nan
    
    # Define daypart ranges
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Daypart'] = df['Time'].apply(assign_daypart)

country_map = {
    'USA': 'USA',
    'U.S.A.': 'USA',
    'US': 'USA',
    'UNITED STATES': 'USA',
    'AUS': 'AUSTRALIA',
    'AUSTRALIA': 'AUSTRALIA'
}
df['Country'] = df['Country'].str.strip().str.upper().replace({
    'USA': 'USA',
    'U.S.A.': 'USA',
    'UNITED STATES': 'USA',
    'AUS': 'AUSTRALIA'
})
# Similar cleaning for 'Area' and 'Location'
df['Area'] = df['Area'].str.strip().str.title()
df['Location'] = df['Location'].str.strip().str.title()

body_parts = ['head', 'eye', 'arm', 'forearm', 'hand', 'finger',
              'leg', 'thigh', 'calf', 'foot', 'toes', 'wrist',
              'torso', 'back', 'chest', 'shin']

injury_types = ['laceration', 'abrasion', 'severed', 'bite', 'nipped', 'puncture', 'cut', 'bruise']

def extract_body_part(injury_text):
    if pd.isna(injury_text):
        return np.nan
    injury_text = injury_text.lower()
    found = [part for part in body_parts if re.search(r'\b' + re.escape(part) + r'\b', injury_text)]
    return ', '.join(found) if found else np.nan

def extract_injury_type(injury_text):
    if pd.isna(injury_text):
        return np.nan
    injury_text = injury_text.lower()
    found = [itype for itype in injury_types if re.search(r'\b' + re.escape(itype) + r'\b', injury_text)]
    return ', '.join(found) if found else np.nan

def assign_severity(row):
    """
    Determine severity based on 'Fatal (Y/N)' and keywords in the 'Injury' description.
    """
    # Convert values to string to safely use strip and case conversion.
    fatal = str(row['Fatal (Y/N)']).strip().upper()
    injury = str(row['Injury']).strip().lower()
    
    if fatal == 'Y':
        return 'Fatal'
    elif any(keyword in injury for keyword in ['minor', 'superficial', 'no injury']):
        return 'Minor'
    elif any(keyword in injury for keyword in ['severe', 'major']):
        return 'Major'
    else:
        return 'Unknown'

df['Body_Part'] = df['Injury'].apply(extract_body_part)
df['Injury_Type'] = df['Injury'].apply(extract_injury_type)
df['Severity'] = df.apply(assign_severity, axis=1)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Month'] = df['Date'].dt.month

def assign_year_group(year):
    """
    Club all years before 1500 into 'Pre1500' and group years from 1500 onward
    into 20-year intervals (e.g., '1500-1519', '1520-1539', etc.).
    """
    try:
        year = int(year)
    except Exception:
        return np.nan
    if year < 1500:
        return "Pre1500"
    else:
        start = (year // 20) * 20
        return f"{start}-{start+19}"
    
def group_sort_key(group_label):
    if group_label == "Pre1500":
        return -1  # Pre1500 comes first
    else:
        return int(group_label.split('-')[0])

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_total = df.dropna(subset=['Year']).copy()
df_total['Year'] = df_total['Year'].astype(int)

df_total['Year_Group'] = df_total['Year'].apply(assign_year_group)

total_by_group = df_total.groupby('Year_Group').size().reset_index(name='Attack_Count')
print("\nTotal Shark Attacks by Year Group:")
print(total_by_group)

total_by_group = total_by_group.sort_values(by='Year_Group', key=lambda col: col.map(group_sort_key))

plt.figure(figsize=(10, 6))
sns.barplot(data=total_by_group, x='Year_Group', y='Attack_Count', palette="viridis")
plt.title('Total Number of Shark Attacks by Year Group')
plt.xlabel('Year Group')
plt.ylabel('Attack Count')
plt.xticks(rotation=45)
plt.show()

activity_counts = df['Activity'].value_counts().reset_index()
activity_counts.columns = ['Activity', 'Count']
print("\nDistribution of Shark Attacks by Activity:")
print(activity_counts.head(10))  # print top 10

plt.figure(figsize=(10, 6))
sns.barplot(data=activity_counts.head(10), x='Activity', y='Count', hue='Activity', palette="rocket")
plt.legend([], [], frameon=False)
plt.title('Top 10 Activities Associated with Shark Attacks')
plt.xlabel('Activity')
plt.ylabel('Attack Count')
plt.xticks(rotation=45)
plt.show()

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

bins = [0, 18, 35, 50, 100]
labels = ['0-18', '19-35', '36-50', '51+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

df['Fatal_Flag'] = df['Fatal (Y/N)'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
fatality_rate = df.groupby(['Country', 'Activity'])['Fatal_Flag'].mean().reset_index()
fatality_rate['Fatality_Rate (%)'] = fatality_rate['Fatal_Flag'] * 100
print("\nFatality Rate by Country and Activity:")
print(fatality_rate.head(10))

df['Species'] = df['Species '].str.strip().str.title()
species_by_region = df.groupby('Country')['Species'].agg(lambda x: x.value_counts().index[0] if not x.isnull().all() else np.nan).reset_index()
species_by_region.columns = ['Country', 'Most_Common_Species']
print("\nMost Common Shark Species by Country:")
print(species_by_region)

features = ['Daypart', 'Activity', 'Species', 'Country']
model_df = df[features + ['Fatal_Flag']].dropna()

for col in features:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))

X = model_df[features]
y = model_df['Fatal_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nPredictive Modeling: Fatality Prediction Report")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if 'Latitude' in df.columns and 'Longitude' in df.columns:
    plt.figure(figsize=(10, 8))
    heat_data = df[['Latitude', 'Longitude']]
    sns.kdeplot(x=heat_data['Longitude'], y=heat_data['Latitude'], shade=True, cmap='Reds')
    plt.title('Heatmap of Shark Attack Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
else:
    print("\nNo latitude/longitude data available for heatmap.")

rare_cases = df[(df['Activity'].str.contains('touching|fishing|tagging', case=False, na=False)) | (df['Injury'].isna())]
print("\nCase Studies of Rare or Unusual Attacks:")
print(rare_cases[['Case Number', 'Activity', 'Injury']].head())

fatal_data = df[df['Fatal_Flag'] == 1].copy()
fatal_data['Year'] = pd.to_numeric(fatal_data['Year'], errors='coerce')
fatal_data = fatal_data.dropna(subset=['Year'])
fatal_data['Year'] = fatal_data['Year'].astype(int)

fatal_data['Year_Group'] = fatal_data['Year'].apply(assign_year_group)

fatal_by_group = fatal_data.groupby('Year_Group').size().reset_index(name='Fatal_Count')

fatal_by_group = fatal_by_group.sort_values(by='Year_Group', key=lambda col: col.map(group_sort_key))

plt.figure(figsize=(12, 6))
ax = sns.lineplot(data=fatal_by_group, x='Year_Group', y='Fatal_Count', marker='o')
plt.title('Fatal Attacks by Year Group (Including Pre1500)')
plt.xlabel('Year Group')
plt.ylabel('Fatal Attack Count')
plt.xticks(rotation=45)

for idx, row in fatal_by_group.iterrows():
    ax.text(row['Year_Group'], row['Fatal_Count'] + 0.5, int(row['Fatal_Count']),
            color='black', ha="center", fontsize=10)

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

fatal_by_month = df[df['Fatal_Flag'] == 1].groupby('Month').size().reset_index(name='Fatal_Count')
print("\nFatal Attacks by Month:")
print(fatal_by_month)

plt.figure(figsize=(8, 5))
sns.barplot(data=fatal_by_month, x='Month', y='Fatal_Count', palette="magma")
plt.title('Fatal Attacks by Month')
plt.xlabel('Month')
plt.ylabel('Fatal Attack Count')
plt.show()

