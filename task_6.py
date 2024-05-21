
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
df = pd.read_csv('user_profiles_for_ads.csv')
 
 # Task Q.1: Import data and check null values, column info, and descriptive statistics of the data
print(df.info())
print(df.isnull().sum())
print(df.describe())
 
 # Task Q.2: Visualizing the distribution of the key demographic variables
 
sns.countplot(x='Age', data=df)
plt.title('Distribution of Age') 
plt.show()
 
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Gender')
plt.show()
 
sns.countplot(x='Education Level', data=df)
plt.title('Distribution of Education Level')
plt.show()
 
sns.countplot(x='Income Level', data=df)
plt.title('Distribution of Income Level')
plt.show()
 
 # Task Q.3: Examine device usage patterns and users' online behaviour
 
sns.countplot(x='Device Usage', data=df)
plt.title('Device Usage Distribution')
plt.show()
 
df[['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']].plot(kind='box')
plt.title('Time Spent Online on Weekdays vs Weekends')
plt.show()
 
 
sns.histplot(df['Likes and Reactions'])
plt.title('Distribution of Likes and Reactions')
plt.show()
 
sns.histplot(df['Followed Accounts'])
plt.title('Distribution of Followed Accounts')
plt.show()
 
sns.histplot(df['Click-Through Rates (CTR)'])
plt.title('Distribution of Click-Through Rates (CTR)')
plt.show()
 
 
sns.histplot(df['Conversion Rates'])
plt.title('Distribution of Conversion Rates')
plt.show()
 
sns.histplot(df['Ad Interaction Time (sec)'])
plt.title('Distribution of Ad Interaction Time (sec)')
plt.show()
 
sns.countplot(y='Top Interests', data=df, order=df['Top Interests'].value_counts().index)
plt.title('Top Interests Distribution')
plt.show()
 
 # Task Q.4: Analyze the average time users spend online on weekdays versus weekends
avg_time_weekday = df['Time Spent Online (hrs/weekday)'].mean()
avg_time_weekend = df['Time Spent Online (hrs/weekend)'].mean()
print(f"Average time spent online on weekdays: {avg_time_weekday} hours")
print(f"Average time spent online on weekends: {avg_time_weekend} hours")
 
 # Task Q.5: Identify the most common interests among users
common_interests = df['Top Interests'].value_counts()
print("Most common interests among users:")
print(common_interests)
 

