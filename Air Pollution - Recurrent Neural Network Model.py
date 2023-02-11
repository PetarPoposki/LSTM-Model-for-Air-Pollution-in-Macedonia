# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:29:57 2023

@author: HP
"""



import glob
import pandas as pd
#import tensorflow as tf
#import keras
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor

def get_percentage_above_threshold(y, threshold):
    return (y > threshold).mean() * 100

# Use glob to get a list of all csv filenames in the current directory
filenames = glob.glob('C:/Users\HP\Desktop\IA Proekt/combined_report_Karpos.csv')

# Create an empty list to store the dataframes
df_list = []

# Loop through the filenames and read each one into a dataframe, then add it to df_list
for file in filenames:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all of the dataframes into a single dataframe
df_combined = pd.concat(df_list)

df_combined['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x))
df_combined = df_combined.sort_values(by='time')
df_combined.reset_index(drop = True, inplace = True)



#ONE TO ONE
# Create new dataframes with the 'PM10', 'PM25', and 'AQI' columns
df_pm10 = pd.DataFrame({'PM10': df_combined['PM10']})
df_pm25 = pd.DataFrame({'PM25': df_combined['PM25']})
df_aqi = pd.DataFrame({'AQI': df_combined['AQI']})
df_time = pd.DataFrame({'time': df_combined['time']})

# Split the data into 70% training data and 30% test data without shuffling
PM10_train, PM10_test = train_test_split(df_pm10, test_size=0.3, shuffle=False)
PM25_train, PM25_test = train_test_split(df_pm25, test_size=0.3, shuffle=False)
AQI_train, AQI_test = train_test_split(df_aqi, test_size=0.3, shuffle=False)



# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# choose a number of time steps
n_steps = 12
# split into samples
X, y = split_sequence(PM10_train['PM10'].values, n_steps)



# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

early_stop = EarlyStopping(monitor='loss', patience=10)

# define model
model = Sequential()
model.add(LSTM(512,input_shape=(n_steps, n_features), activation='relu',return_sequences=True))
model.add(LSTM(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adamax', loss='mse')

# fit model
model.fit(X, y, epochs=25, batch_size=128, verbose=1, callbacks=[early_stop])


# Split the test data into the same number of time steps as the training data
X_test, y_test = split_sequence(PM10_test['PM10'].values, n_steps)

# Reshape the test data into the expected input shape
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# Get predictions on the test set
test_predictions = model.predict(X_test)



N = min(len(PM10_test), len(test_predictions))

mse = mean_squared_error(PM10_test[:N], test_predictions[:N])

# Calculate the mean squared error (MSE) and root mean squared error (RMSE)
mse = mean_squared_error(PM10_test[:N], test_predictions[:N])
rmse = np.sqrt(mse)

# Calculate the mean absolute error (MAE)
mae = mean_absolute_error(PM10_test[:N], test_predictions[:N])

# Calculate the R^2 score
r2 = r2_score(PM10_test[:N], test_predictions[:N])

# Print the results
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("Mean Absolute Error: ", mae)
print("R^2 Score: ", r2)





#MANY TO ONE

#PM10
early_stop = EarlyStopping(monitor='loss', patience=8)
df_combined = df_combined.select_dtypes(include=[np.number])
df_combined.fillna(df_combined.median(), inplace=True)


# Setting up the dependent variable and independent variables
X = df_combined.drop("PM10", axis=1)
y = df_combined["PM10"]




# Splitting data into training and testing sets
train_size = int(0.7 * len(df_combined))
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

train_threshold = get_percentage_above_threshold(y_train, 175)
test_threshold = get_percentage_above_threshold(y_test, 175)

print("Percentage of PM10 values above 175 in the train set: {:.2f}%".format(train_threshold))
print("Percentage of PM10 values above 175 in the test set: {:.2f}%".format(test_threshold))



# Reshaping data for the LSTM model
timesteps = 12
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = np.array([X_train[i:i + timesteps] for i in range(len(X_train) - timesteps + 1)])
X_test = np.array([X_test[i:i + timesteps] for i in range(len(X_test) - timesteps + 1)])
y_train = y_train[timesteps - 1:]
y_test = y_test[timesteps - 1:]



# Defining the LSTM model
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adamax')

# Training the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, callbacks=[early_stop])

# Making predictions on the test set
predictions = model.predict(X_test)

# Calculating metrics on the test set
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)



#Display actual vs predicted for a certain month
#Specific values are used which won't work with every dataset
import datetime
import matplotlib.dates as mdates
dates = []
dates = df_combined['time'][21711:21978]

probno = y_test[6231:6498]
probno1 = predictions[6231:6498]


fig, ax = plt.subplots()
ax.plot(dates, probno, label='Actual PM10')
ax.plot(dates, probno1, label='Predicted PM10')

ax.set_xlabel('Date')
ax.set_ylabel('PM10 Value')
ax.legend()
# format the dates to display only the month and year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# set the number of x-ticks and their locations
num_ticks = 6  # set the number of x-ticks
spacing = int(len(dates) / num_ticks)  # calculate the tick spacing
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=spacing))

plt.tight_layout()

plt.show()




# Plotting the actual vs predicted values
plt.plot(y_test, label="Actual PM10 Values")
plt.plot(predictions, label="Predicted PM10 Values")
plt.title("Kumanovo - MSE loss function")
plt.legend()
plt.show


# Plotting the actual vs predicted values
plt.scatter(y_test, predictions, c='red', s=5, label="Predicted PM10 Values")
plt.scatter(y_test, y_test, c='blue', s=5, label="Actual PM10 Values")
plt.xlabel("Actual PM10 Values")
plt.ylabel("Predicted PM10 Values")
plt.title("Kumanovo - MSE loss Function")
plt.legend()
plt.show()







# Defining the Dummy Regressor
dummy_regr = DummyRegressor()

# Training the Dummy Regressor
dummy_regr.fit(X_train, y_train)

# Making predictions on the test set
dummy_predictions = dummy_regr.predict(X_test)

# Calculating metrics on the test set
mae = mean_absolute_error(y_test, dummy_predictions)
rmse = np.sqrt(mean_squared_error(y_test, dummy_predictions))
r2 = r2_score(y_test, dummy_predictions)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Plotting the actual vs predicted values
plt.scatter(y_test, dummy_predictions, c='red', label="Predicted PM10 Values")
plt.scatter(y_test, y_test, c='blue', label="Actual PM10 Values")
plt.xlabel("Actual PM10 Values")
plt.ylabel("Predicted PM10 Values")
plt.title("Kavadarci")
plt.legend()
plt.show()











#PM2.5

early_stop = EarlyStopping(monitor='loss', patience=8)
df_combined = df_combined.select_dtypes(include=[np.number])
df_combined.fillna(df_combined.median(), inplace=True)



# Setting up the dependent variable and independent variables
X = df_combined.drop("PM25", axis=1)
y = df_combined["PM25"]



# Splitting data into training and testing sets
train_size = int(0.7 * len(df_combined))
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

train_threshold = get_percentage_above_threshold(y_train, 200)
test_threshold = get_percentage_above_threshold(y_test, 200)

print("Percentage of PM25 values above 200 in the train set: {:.2f}%".format(train_threshold))
print("Percentage of PM25 values above 200 in the test set: {:.2f}%".format(test_threshold))



# Reshaping data for the LSTM model
timesteps = 12
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = np.array([X_train[i:i + timesteps] for i in range(len(X_train) - timesteps + 1)])
X_test = np.array([X_test[i:i + timesteps] for i in range(len(X_test) - timesteps + 1)])
y_train = y_train[timesteps - 1:]
y_test = y_test[timesteps - 1:]



# Defining the LSTM model
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss="mae", optimizer='adamax')

# Training the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, callbacks=[early_stop])

# Making predictions on the test set
predictions = model.predict(X_test)


# Calculating metrics on the test set
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Plotting the actual vs predicted values
plt.plot(y_test, label="Actual PM25 Values")
plt.plot(predictions, label="Predicted PM25 Values")
plt.title("Karposh")
plt.legend()
plt.show


# Plotting the actual vs predicted values
plt.scatter(y_test, predictions, c='red', label="Predicted PM25 Values")
plt.scatter(y_test, y_test, c='blue', label="Actual PM25 Values")
plt.xlabel("Actual PM25 Values")
plt.ylabel("Predicted PM25 Values")
plt.title("Karposh")
plt.legend()
plt.show()





