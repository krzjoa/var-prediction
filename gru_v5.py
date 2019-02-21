from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Input, Concatenate, Lambda, Conv1D
import keras.backend as K
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import numpy as np
np.random.seed(7)

from tensorflow import set_random_seed
set_random_seed(7)

import os
import pandas as pd

# Params
MODEL_NAME = "gru_v5_15e"
EPOCHS = 10
BATCH_SIZE = 72
OUTPUT = "output"
SUBMISSIONS = "submissions"

# Loading data
loaded = np.load("data/rnn_input_s21_minmax.npz")
X = loaded['X_train']

X_test_holdout = loaded['X_test']

y = loaded['y_train']
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
conv = Conv1D(32, 3)(inputs)
conv = Conv1D(64, 3)(conv)
forward = GRU(50, activation='elu')(conv)
backward = GRU(50, go_backwards=True, activation='elu')(conv)
#forward = Lambda(lambda x: x[:, 11, :])(forward)
#backward = Lambda(lambda x: x[:, 11, :])(backward)
net = Concatenate()([forward, backward])
net = Dense(10, activation='relu')(net)
net = Dense(1, activation='linear')(net)


model = Model(inputs=inputs, outputs=net)
model.compile(optimizer='RMSProp', loss='mae')


# Training
history = History()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          validation_split=0.2,
          #validation_data=[X_val, y_val],
          shuffle=True,
          epochs=EPOCHS,
          callbacks=[history])


y_pred = model.predict(X_test)


# Get values back
#from sklearn.externals import joblib
#var_scaler = joblib.load('data/var_scaler.pkl')
#y_pred = var_scaler.inverse_transform(y_pred)

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

#
y_holdout_pred  = model.predict(X_test_holdout[:96]).ravel()
y_holdout_dates = range(96)
pd.DataFrame({"Date": y_holdout_dates, "VAR": y_holdout_pred}).to_csv(
    os.path.join(SUBMISSIONS, "gru_v5.csv"), index=None)