import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('attacks.xlsx')

print(data.head())
print(data.tail())

# getting statistical values
print(data.describe())

# checking data types
print(data.info())

# checking null values 
print(data.isnull().sum())


