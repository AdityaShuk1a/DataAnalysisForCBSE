import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\DataSetAnalysis\shark data\attacks.csv", encoding='latin-1')
# Performing EDA
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())

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

# 1.2 Standardize Inconsistent Country/Location Names
# Create a mapping dictionary for common inconsistencies (adjust as needed)
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

# Define a helper function for sorting the groups
def group_sort_key(group_label):
    if group_label == "Pre1500":
        return -1  # Pre1500 comes first
    else:
        return int(group_label.split('-')[0])

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_total = df.dropna(subset=['Year']).copy()
df_total['Year'] = df_total['Year'].astype(int)

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