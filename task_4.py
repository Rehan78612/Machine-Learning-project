import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('userbehaviour.csv')

# Q.1: Check null values, column info, and descriptive statistics
print("Null values:")
print(data.isnull().sum())
print("\nColumn info:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())

# Q.2: Check the highest, lowest, and average screen time
print("\nScreen time:")
print("Max: ", data['Average Screen Time'].max())
print("Min: ", data['Average Screen Time'].min())
print("Mean: ", data['Average Screen Time'].mean())

# Q.3: Check the highest, lowest, and the average amount spent
print("\nAmount spent:")
print("Max: ", data['Average Spent on App (INR)'].max())
print("Min: ", data['Average Spent on App (INR)'].min())
print("Mean: ", data['Average Spent on App (INR)'].mean())

# Query 4: Correlation between 'Average Screen Time' and 'Average Spent on App (INR)'

correlation_matrix = data[['Average Screen Time', 'Average Spent on App (INR)']].corr()
correlation_value = correlation_matrix.loc['Average Screen Time', 'Average Spent on App (INR)']
print("Correlation between 'Average Screen Time' and 'Average Spent on App (INR)':")
print(correlation_value)

# Query 5: Correlation between 'Ratings' and other variables

correlation_matrix = data[['Ratings', 'Average Screen Time', 'Average Spent on App (INR)']].corr()
correlation_with_screen_time = correlation_matrix.loc['Ratings', 'Average Screen Time']
correlation_with_spent = correlation_matrix.loc['Ratings', 'Average Spent on App (INR)']

print("Correlation between 'Ratings' and 'Average Screen Time':")
print(correlation_with_screen_time)
print("Correlation between 'Ratings' and 'Average Spent on App (INR)':")
print(correlation_with_spent)

# Q.6: App User segmentation using K-means clustering

X = data[['Average Screen Time', 'Average Spent on App (INR)', 'Ratings']]
kmeans = KMeans(n_clusters=3, random_state=42)  # Setting random_state for reproducibility
kmeans.fit(X)
data['Cluster'] = kmeans.labels_

# Q.7: Visualize the segments
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', hue='Cluster', data=data, palette='viridis')
plt.title('User Segments based on Screen Time and Amount Spent')
plt.xlabel('Average Screen Time')
plt.ylabel('Average Spent on App (INR)')
plt.legend(title='Cluster')
plt.show()

# Q.8: Explain the summary of your working
summary = """
Summary of Work:
1. Data Inspection:
   - Checked for null values, column information, and descriptive statistics of the dataset.
   - Identified the maximum, minimum, and mean values for 'Average Screen Time' and 'Average Spent on App (INR)'.

2. Correlation Analysis:
   - Examined the correlation between 'Average Screen Time', 'Average Spent on App (INR)', and 'Ratings'.
   - Found that there are correlations indicating relationships between these variables which can be further analyzed.

3. Clustering Analysis:
   - Applied K-Means clustering to segment users into 3 clusters based on their 'Average Screen Time', 'Average Spent on App (INR)', and 'Ratings'.
   - Visualized these segments using a scatter plot.

4. Findings and Recommendations:
   - The clustering revealed distinct user segments with varying screen time, spending behavior, and ratings.
   - Further analysis could be done to tailor marketing strategies, app features, or content to different user segments to improve user engagement and revenue.
"""
print(summary)
