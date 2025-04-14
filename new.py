# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load dataset (Example: user-course interaction data)
data = pd.read_csv("user_course_data.csv")

# Data Preprocessing
data = data.dropna()  # Handling missing data

# Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of ratings
sns.histplot(data['rating'], bins=10)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Content-based Recommender System: Using Course Genre & User Profile
course_features = data[['course_id', 'course_genre', 'difficulty']]
user_profile = data[['user_id', 'user_preferences']]

# Creating a simple content-based recommendation
def content_based_recommender(user_id):
    user_data = user_profile[user_profile['user_id'] == user_id]
    recommended_courses = course_features[course_features['course_genre'] == user_data['user_preferences'].values[0]]
    return recommended_courses

# KNN-based Collaborative Filtering
X = data[['user_id', 'course_id']]  # Assuming we have user-course interaction data
y = data['rating']

knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X)
recommended_courses_knn = knn.kneighbors([X.iloc[0]], n_neighbors=5)

# NMF-based Collaborative Filtering
nmf = NMF(n_components=20, random_state=1)
W = nmf.fit_transform(data.drop(columns=['user_id', 'course_id', 'rating']))
H = nmf.components_

# Neural Network Embedding-based Collaborative Filtering (using TensorFlow)
input_user = tf.keras.layers.Input(shape=(1,))
input_item = tf.keras.layers.Input(shape=(1,))

embedding_user = tf.keras.layers.Embedding(input_dim=len(data['user_id'].unique()), output_dim=50)(input_user)
embedding_item = tf.keras.layers.Embedding(input_dim=len(data['course_id'].unique()), output_dim=50)(input_item)

dot_product = tf.keras.layers.Dot(axes=1)([embedding_user, embedding_item])
model = tf.keras.Model(inputs=[input_user, input_item], outputs=dot_product)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit([data['user_id'], data['course_id']], data['rating'], epochs=10)

# Performance Evaluation
y_pred = model.predict([data['user_id'], data['course_id']])
mse = mean_squared_error(data['rating'], y_pred)
print(f"Mean Squared Error: {mse}")
