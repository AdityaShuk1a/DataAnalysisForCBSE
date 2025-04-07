import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\DataSetAnalysis\shark data\attacks.csv", encoding='latin-1')
# Display first few rows to verify columns and data
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_total = df.dropna(subset=['Year']).copy()
df_total['Year'] = df_total['Year'].astype(int)