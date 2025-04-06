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
# print(data.isnull().sum())
months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 
          'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

#filttering data for years which are not in the correct form
# data = data[data["Year"].str.contains("Reported")]
# data["Year"] = data["Year"].str.replace("Reported", "")


date = data["Date"]
#fixing dates
# for i, item in enumerate(date):
#     if str(item) != "" and len(str(item)) > 6:
#         print(data.at[i, 'Date']) 

count = 0

for i, item in enumerate(date):
    strItem = str(data.at[i, 'Date'])
    """
    This loop is used to fix the dates which are not in the correct form 
    and erase the non digit values
    """
    
    valTillSpaceOrNotDigit = 0
    for j in range(len(str(item))):
        val = str(item)
        
        if val[j].isdigit() == False:
            valTillSpaceOrNotDigit = j+1
        else:
            break
   
    item = strItem[valTillSpaceOrNotDigit:]
    strItem = strItem[valTillSpaceOrNotDigit:]
    
    if strItem[3:6] in months:
        strItem = strItem[:3] + months[strItem[3:6]] + strItem[6:]
        
    print(strItem)
    count+=1
    
print(count)
        
    
    

