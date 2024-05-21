# Q.1: Import data and check null values, column info, and descriptive statistics of the data.

# Q.2: Convert the Date column into datetime datatype to move forward.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
data = pd.read_csv("Instagram-Reach.csv")

print("Null Values:")
print(data.isnull().sum())
print("\nColumn Info:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())

data['Date'] = pd.to_datetime(data['Date'])

print("\nFirst few rows after conversion:")
print(data.head())

# Q.3: Analyze the trend of Instagram reach over time using a line chart


plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Instagram reach'], marker='o', linestyle='-')
plt.title('Instagram Reach Over Time')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Q.4: Analyze Instagram reach for each day using a bar chart.

plt.figure(figsize=(14, 6))
plt.bar(data['Date'], data['Instagram reach'], color='skyblue')
plt.title('Instagram Reach for Each Day')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Q.5: Analyze the distribution of Instagram reach using a box plot

plt.figure(figsize=(8, 6))
plt.boxplot(data['Instagram reach'], vert=False)
plt.title('Distribution of Instagram Reach')
plt.xlabel('Instagram Reach')
plt.yticks([])
plt.grid(True)
plt.tight_layout()
plt.show()

# Q.6: Analyze the reach based on the days of the week.


data['Day'] = data['Date'].dt.day_name()
reach_stats_by_day = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std'])
print(reach_stats_by_day)

# Q.7: Create a bar chart to visualize the reach for each day of the week.

plt.figure(figsize=(10, 6))
data.groupby('Day')['Instagram reach'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Instagram Reach by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Instagram Reach')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Q.8: Check the Trends and Seasonal patterns of Instagram reach.

decomposition = seasonal_decompose(data['Instagram reach'], model='additive', period=7)  # assuming weekly seasonality
decomposition.plot()
plt.show()

# Q.9: Visualize an autocorrelation plot to find the value of p and partial autocorrelation plot to find the value of q.

plot_acf(data['Instagram reach'], lags=40)
plt.show()
plot_pacf(data['Instagram reach'], lags=40)
plt.show()

# Q.10: Train a model using SARIMA and make predictions.

train_data = data[data['Date'] < '2022-10-01']
test_data = data[data['Date'] > '2022-10-01']

print(test_data)





