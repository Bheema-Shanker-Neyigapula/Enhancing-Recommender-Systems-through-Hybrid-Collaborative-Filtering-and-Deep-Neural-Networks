import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import Dense, Concatenate, Input
from keras.models import Model

# Sample data (replace this with your actual dataset)
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'item_id': [101, 102, 101, 103, 102, 104, 105, 104],
    'rating': [5, 4, 3, 4, 5, 3, 4, 3],
}

df = pd.DataFrame(data)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Collaborative Filtering
def collaborative_filtering(train_data, user_id, item_id):
    user_ratings = train_data[train_data['user_id'] == user_id]
    item_ratings = train_data[train_data['item_id'] == item_id]

    if user_ratings.empty or item_ratings.empty:
        return 0  # No collaborative filtering prediction available

    user_mean = user_ratings['rating'].mean()
    item_mean = item_ratings['rating'].mean()

    return (user_mean + item_mean) / 2

# User-Based Collaborative Filtering
def user_based_cf(train_data, user_id, item_id, k=3):
    user_ratings = train_data[train_data['user_id'] == user_id].drop(columns=['user_id'])
    user_ratings.set_index('item_id', inplace=True)
    
    if item_id not in user_ratings.index:
        return 0  # No collaborative filtering prediction available for the item
    
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nn.fit(train_data.drop(columns=['user_id']))
    distances, indices = nn.kneighbors([user_ratings.loc[item_id].values])
    neighbor_ratings = train_data.iloc[indices[0]]
    return neighbor_ratings['rating'].mean()

# Item-Based Collaborative Filtering
def item_based_cf(train_data, user_id, item_id, k=3):
    item_ratings = train_data[train_data['item_id'] == item_id].drop(columns=['item_id'])
    item_ratings.set_index('user_id', inplace=True)
    
    if user_id not in item_ratings.index:
        return 0  # No collaborative filtering prediction available for the user
    
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nn.fit(train_data.drop(columns=['item_id']))
    distances, indices = nn.kneighbors([item_ratings.loc[user_id].values])
    neighbor_ratings = train_data.iloc[indices[0]]
    return neighbor_ratings['rating'].mean()

# Neural Network Model
def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare data for neural network
X_train = train_df[['user_id', 'item_id']]
y_train = train_df['rating']
X_test = test_df[['user_id', 'item_id']]
y_test = test_df['rating']

# Collaborative Filtering predictions
train_df['cf_predictions'] = train_df.apply(lambda row: collaborative_filtering(train_df, row['user_id'], row['item_id']), axis=1)
test_df['cf_predictions'] = test_df.apply(lambda row: collaborative_filtering(train_df, row['user_id'], row['item_id']), axis=1)

# User-Based CF predictions
test_df['user_based_cf_predictions'] = test_df.apply(lambda row: user_based_cf(train_df, row['user_id'], row['item_id']), axis=1)

# Item-Based CF predictions
test_df['item_based_cf_predictions'] = test_df.apply(lambda row: item_based_cf(train_df, row['user_id'], row['item_id']), axis=1)

# Neural Network predictions
input_dim = X_train.shape[1]
nn_model = build_nn_model(input_dim)
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
test_df['nn_predictions'] = nn_model.predict(X_test)

# Hyperparameter analysis for User-Based Collaborative Filtering
k_values = [1, 3, 5, 7, 10]
rmse_user_cf_values = []

for k in k_values:
    test_df[f'user_based_cf_predictions_{k}'] = test_df.apply(lambda row: user_based_cf(train_df, row['user_id'], row['item_id'], k=k), axis=1)
    rmse_user_cf = np.sqrt(mean_squared_error(y_test, test_df[f'user_based_cf_predictions_{k}']))
    rmse_user_cf_values.append(rmse_user_cf)

# Hyperparameter analysis for Neural Network
epochs_values = [5, 10, 15, 20]
rmse_nn_values = []

for epochs in epochs_values:
    nn_model = build_nn_model(input_dim)
    nn_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    test_df[f'nn_predictions_{epochs}'] = nn_model.predict(X_test)
    rmse_nn = np.sqrt(mean_squared_error(y_test, test_df[f'nn_predictions_{epochs}']))
    rmse_nn_values.append(rmse_nn)

# Hybrid Model: Combining Collaborative Filtering and Neural Network
# Create a new input layer for collaborative filtering predictions
cf_input = Input(shape=(1,))

# Concatenate the CF predictions and NN predictions as inputs to the final layer
combined = Concatenate()([cf_input, nn_model.output])

# Create the final output layer
hybrid_output = Dense(1, activation='linear')(combined)

# Create the hybrid model
hybrid_model = Model(inputs=[cf_input, nn_model.input], outputs=hybrid_output)

# Compile and fit the hybrid model
hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
hybrid_model.fit([train_df['cf_predictions'], X_train], y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions using the hybrid model
test_df['hybrid_predictions'] = hybrid_model.predict([test_df['cf_predictions'], X_test])

# Calculate RMSE for each model
rmse_cf = np.sqrt(mean_squared_error(y_test, test_df['cf_predictions']))
rmse_user_cf = np.sqrt(mean_squared_error(y_test, test_df['user_based_cf_predictions']))
rmse_item_cf = np.sqrt(mean_squared_error(y_test, test_df['item_based_cf_predictions']))
rmse_nn = np.sqrt(mean_squared_error(y_test, test_df['nn_predictions']))
rmse_hybrid = np.sqrt(mean_squared_error(y_test, test_df['hybrid_predictions']))

# Print RMSE values
print("RMSE for Collaborative Filtering:", rmse_cf)
print("RMSE for User-Based Collaborative Filtering:", rmse_user_cf)
print("RMSE for Item-Based Collaborative Filtering:", rmse_item_cf)
print("RMSE for Neural Network Model:", rmse_nn)
print("RMSE for Hybrid Recommender:", rmse_hybrid)

# Plot RMSE values
models = ['Collaborative Filtering', 'User-Based CF', 'Item-Based CF', 'Neural Network', 'Hybrid']
rmse_values = [rmse_cf, rmse_user_cf, rmse_item_cf, rmse_nn, rmse_hybrid]

plt.figure(figsize=(8, 6))
plt.bar(models, rmse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Models')
plt.ylim(0, max(rmse_values) + 0.5)
plt.show()

# Hyperparameter analysis for Hybrid Model
rmse_hybrid_values = []
for k in k_values:
    for epochs in epochs_values:
        hybrid_model = Model(inputs=[cf_input, nn_model.input], outputs=hybrid_output)
        hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
        hybrid_model.fit([train_df['cf_predictions'], X_train], y_train, epochs=epochs, batch_size=32, verbose=0)
        test_df[f'hybrid_predictions_{k}_{epochs}'] = hybrid_model.predict([test_df[f'user_based_cf_predictions_{k}'], X_test])
        rmse_hybrid = np.sqrt(mean_squared_error(y_test, test_df[f'hybrid_predictions_{k}_{epochs}']))
        rmse_hybrid_values.append(rmse_hybrid)

# Plotting the results
plt.figure(figsize=(12, 6))

# User-Based CF hyperparameter analysis
plt.subplot(1, 3, 1)
plt.plot(k_values, rmse_user_cf_values, marker='o')
plt.xlabel('Number of Neighbors (k) in User-Based CF')
plt.ylabel('RMSE')
plt.title('User-Based CF Hyperparameter Analysis')

# Neural Network hyperparameter analysis
plt.subplot(1, 3, 2)
plt.plot(epochs_values, rmse_nn_values, marker='o')
plt.xlabel('Number of Epochs in Neural Network')
plt.ylabel('RMSE')
plt.title('Neural Network Hyperparameter Analysis')

# Hybrid Model hyperparameter analysis
plt.subplot(1, 3, 3)
for i, k in enumerate(k_values):
    plt.plot(epochs_values, rmse_hybrid_values[i*len(epochs_values):(i+1)*len(epochs_values)], marker='o', label=f'k={k}')
plt.xlabel('Number of Epochs in Neural Network')
plt.ylabel('RMSE')
plt.title('Hybrid Model Hyperparameter Analysis')
plt.legend()

plt.tight_layout()
plt.show()