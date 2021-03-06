from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Input, Concatenate
import keras.backend as K
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import numpy as np
np.random.seed(7)

from tensorflow import set_random_seed
set_random_seed(7)

import os

# Params
MODEL_NAME = "gru_v4_10e"
EPOCHS = 50
BATCH_SIZE = 64
OUTPUT = "output"


# Loading data
loaded = np.load("data/rnn_input_s21_minmax.npz")
X = loaded['X_train']

X_test_holdout = loaded['X_test']

y = loaded['y_train_scaled']
y_orig = loaded['y_train']

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
y_test_orig = y_orig[train_dataset_last_idx:]

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Model
inputs = Input((21, 9))
forward = GRU(30, activation='relu')(inputs)
backward = GRU(30, go_backwards=True, activation='relu')(inputs)
net = Concatenate()([forward, backward])
net = Dense(1)(net)


model = Model(inputs=inputs, outputs=net)
model.compile(optimizer='adam', loss='mse')


# Training
history = History()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          validation_split=0.2,
          #validation_data=[X_val, y_val],
          epochs=EPOCHS,
          callbacks=[history])


y_pred = model.predict(X_test)


# Get values back
from sklearn.externals import joblib
var_scaler = joblib.load('data/var_scaler.pkl')
y_pred = var_scaler.inverse_transform(y_pred)

rmse = np.sqrt(((y_test_orig - y_pred)**2).mean())
print "RMSE:", rmse

mae = np.abs(y_test_orig - y_pred).mean()
print "MAE:", mae

y_test_orig = y_test_orig.ravel()
y_pred = y_pred.ravel()

y_mean = y_test.mean()
ss_residual = ((y_test_orig - y_pred)**2).sum()
ss_total = ((y_test_orig - y_mean)**2).sum()
R2 = 1 - (ss_residual / ss_total)
print "R^2:", R2

# Results
plt.plot(y_test_orig)
plt.plot(y_pred)
plt.suptitle("Results")
plt.savefig(os.path.join(OUTPUT, MODEL_NAME+"_results.png"))
plt.close()


# Plot loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.suptitle("Loss history")
plt.savefig(os.path.join(OUTPUT, MODEL_NAME+".png"))

print history.history