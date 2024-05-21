
# Step 1: Import necessary libraries and load the data

import pandas as pd
from collections import Counter
import re
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import IsolationForest

pio.templates.default = "plotly_white"

data = pd.read_csv('Queries.csv')
print(data.head())
# Step 2: Check for null values, column info, and descriptive statistics

print(data.isnull().sum())

print(data.info())

print(data.describe())

# Step 3: Convert the CTR column from a percentage string to a float

data['CTR'] = data['CTR'].str.rstrip('%').astype('float') / 100.0

print(data['CTR'].head())

#  Step 4: Analyze common words in each search query
def clean_and_split(query):
    return re.findall(r'\b\w+\b', query.lower())

all_words = Counter()

data['Top queries'].apply(lambda x: all_words.update(clean_and_split(x)))

word_freq = pd.DataFrame(all_words.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

fig = px.bar(word_freq.head(20), x='Word', y='Frequency', title='Top 20 Words in Search Queries')
fig.show()

# Step 5: Top queries by clicks and impressions

top_clicks = data.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
print(top_clicks)

top_impressions = data.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]
print(top_impressions)

# Step 6: Analyze queries with the highest and lowest CTRs

highest_ctr = data.nlargest(10, 'CTR')[['Top queries', 'CTR']]
print(highest_ctr)

lowest_ctr = data.nsmallest(10, 'CTR')[['Top queries', 'CTR']]
print(lowest_ctr)

# Step 7: Check the correlation between different metrics 

correlation_matrix = data[['Clicks', 'Impressions', 'CTR', 'Position']].corr()

fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
fig.show()

print(correlation_matrix)

# Step 8: Detect anomalies in search queries using Isolation Forest

features = data[['Clicks', 'Impressions', 'CTR', 'Position']]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(features)

anomalies = data[data['Anomaly'] == -1]
print(anomalies)


