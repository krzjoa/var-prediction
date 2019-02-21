from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
from keras.callbacks import History
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(7)

from tensorflow import set_random_seed
set_random_seed(7)

import os

# Params
EPOCHS = 10
BATCH_SIZE = 32
OUTPUT = "output"


# Loading data
loaded = np.load("data/rnn_input.npz")
X = loaded['X_train']

X_test_holdout = loaded['X_test']

y = loaded['y_train']
#y_test = loaded['y_test']


# Get validation dataset
train_dataset_last_idx = int(X.shape[0] * 0.8)
#test_dataset_last_idx = int(X.shape[0] * 0.8)

#print train_dataset_last_idx, test_dataset_last_idx

X_train = X[:train_dataset_last_idx]
y_train = y[:train_dataset_last_idx]

#X_test = X[train_dataset_last_idx:test_dataset_last_idx]
#y_test = y[train_dataset_last_idx:test_dataset_last_idx]

X_test = X[train_dataset_last_idx:]
y_test = y[train_dataset_last_idx:]

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(13, 11)))
model.add(Dense(1))
model.compile(optimizer='adam', loss=rmse)


# Training
history = History()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          validation_split=0.2,
          #validation_data=[X_val, y_val],
          epochs=EPOCHS,
          callbacks=[history])


y_pred = model.predict(X_test)

rmse = np.sqrt(((y_test - y_pred)**2).mean())
print "RMSE:", rmse

mae = np.abs(y_test - y_pred).mean()
print "MAE:", mae

y_test = y_test.ravel()
y_pred = y_pred.ravel()

y_mean = y_test.mean()
ss_residual = ((y_test - y_pred)**2).sum()
ss_total = ((y_test - y_mean)**2).sum()
R2 = 1 - (ss_residual / ss_total)
print "R^2:", R2

# Results
plt.plot(y_test)
plt.plot(y_pred)
plt.suptitle("Results")
plt.savefig(os.path.join(OUTPUT, "lstm_results.png"))
plt.close()


# Plot loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.suptitle("Loss history")
plt.savefig(os.path.join(OUTPUT, "lstm_50.png"))

print history.history