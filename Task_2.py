# Query 1: Import data and check null values, column info, and descriptive statistics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
tips = pd.read_csv('tips.csv')
print(tips.isnull().sum())
print(tips.info())
print(tips.describe())

# Query 2: Tips analysis based on total bill, number of people, and day of the week



sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Tips vs Total Bill')
plt.show()

sns.scatterplot(x='size', y='tip', data=tips)
plt.title('Tips vs Number of People')
plt.show()

sns.boxplot(x='day', y='tip', data=tips)
plt.title('Tips vs Day of the Week')
plt.show()

# Query 3: Tips analysis based on total bill, number of people, and gender of the person paying the bill
# python

sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)
plt.title('Tips vs Total Bill by Gender')
plt.show()

sns.scatterplot(x='size', y='tip', hue='sex', data=tips)
plt.title('Tips vs Number of People by Gender')
plt.show()

# Query 4: Tips analysis based on total bill, number of people, and time of the meal

sns.scatterplot(x='total_bill', y='tip', hue='time', data=tips)
plt.title('Tips vs Total Bill by Time')
plt.show()


sns.scatterplot(x='size', y='tip', hue='time', data=tips)
plt.title('Tips vs Number of People by Time')
plt.show()

# Query 5: Tips by day of the week

sns.boxplot(x='day', y='tip', data=tips)
plt.title('Tips vs Day of the Week')
plt.show()

# Query 6: Tips by gender of the person paying the bill

sns.boxplot(x='sex', y='tip', data=tips)
plt.title('Tips by Gender')
plt.show()

# Query 7: Tips by smoker or non-smoker

sns.boxplot(x='smoker', y='tip', data=tips)
plt.title('Tips by Smoker vs Non-Smoker')
plt.show()

# Query 8: Tips during lunch or dinner

sns.boxplot(x='time', y='tip', data=tips)
plt.title('Tips during Lunch vs Dinner')
plt.show()

# Query 9: Transform categorical values into numerical values

tips['sex'] = tips['sex'].map({'Female': 0, 'Male': 1})
tips['smoker'] = tips['smoker'].map({'No': 0, 'Yes': 1})
tips['day'] = tips['day'].map({'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3})
tips['time'] = tips['time'].map({'Lunch': 0, 'Dinner': 1})

print(tips.head())

# Query 10: Train a Linear Regression model for waiter tips prediction

X = tips[['total_bill', 'sex', 'smoker', 'day', 'time', 'size']]
y = tips['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Query 11: Model prediction for given input

input_data = {'total_bill': 24.50, 'sex': 1, 'smoker': 0, 'day': 0, 'time': 1, 'size': 4}
input_df = pd.DataFrame([input_data])

predicted_tip = model.predict(input_df)
print(f'Predicted Tip: {predicted_tip}')





