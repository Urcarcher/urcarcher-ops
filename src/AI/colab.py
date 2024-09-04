# !pip install numpy
# !pip install pandas
# !pip install tensorflow==2.15.1
# !pip install keras==2.15.0
# !pip install scikit-learn
# !pip install matplotlib

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

TRAIN_PATH = "drive/MyDrive/urcarcher-ml"
TRAIN_TARGET = "USD_KRW"
COLUMNS = ["Date", "Price", "Open", "High", "Low", "Change %"]
COLUMNS_MATCHING = {"Price":0, "Open":1, "High":2, "Low":3}

PREDICT_TARGET = "Open"

exchange_rate_data = pd.read_csv(f"{TRAIN_PATH}/{TRAIN_TARGET}.csv")

y = exchange_rate_data[PREDICT_TARGET].values

dates = pd.to_datetime(exchange_rate_data["Date"])

exchange_rate_data = exchange_rate_data[COLUMNS[1:]].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(exchange_rate_data)
exchange_rate_data_scaled = scaler.transform(exchange_rate_data)

# n_train = int(0.95 * exchange_rate_data_scaled.shape[0])
n_train = exchange_rate_data_scaled.shape[0]
train_data_scaled = exchange_rate_data_scaled[:n_train]
train_dates = dates[:n_train]

test_data_scaled = exchange_rate_data_scaled[n_train:]
test_dates = dates[n_train:]

pred_days = 260
seq_len = 1300
input_dim = 1

trainX, trainY = [], []

# for i in range(seq_len, n_train-pred_days):
#     trainX.append(train_data_scaled[i-seq_len:i, 0:train_data_scaled.shape[1]])
#     trainY.append(train_data_scaled[i:i+pred_days, 0])

# for i in range(seq_len, len(test_data_scaled)-pred_days+1) :
#     testX.append(test_data_scaled[i-seq_len:i, 0:test_data_scaled.shape[1]])
#     testY.append(test_data_scaled[i:i+pred_days, 0])

for i in range(seq_len, n_train-pred_days, seq_len):
    trainX.append(train_data_scaled[i-seq_len:i, COLUMNS_MATCHING[PREDICT_TARGET]])
    trainY.append(train_data_scaled[i:i+pred_days, COLUMNS_MATCHING[PREDICT_TARGET]])

trainX, trainY = np.array(trainX), np.array(trainY)
early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)

# input_shape = (trainX.shape[1], trainX.shape[2])
input_shape = (seq_len, 1)

MODEL_NAME = "BiLSTM"
WEIGHTS_USING = f'{TRAIN_PATH}/weights/{MODEL_NAME}_p{pred_days}_s{seq_len}_d{input_dim}_{TRAIN_TARGET}_{PREDICT_TARGET}.weights.h5'

# BiLSTM V
model = Sequential()
model.add(Bidirectional(LSTM(units = 512,
               input_shape = input_shape,
               return_sequences = True,
               )))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(units = 512,
               return_sequences = True,
               )))
model.add(Dropout(0.4))
model.add(GRU(units = 256,
               return_sequences = True
               ))
model.add(GRU(units = 256,
               return_sequences = False
               ))
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Dense(trainY.shape[1]))

# BiLSTM_T V
# model = Sequential()
# model.add(Bidirectional(LSTM(units = 512,
#                input_shape = input_shape,
#                return_sequences = True,
#                )))
# model.add(Dropout(0.4))
# model.add(Bidirectional(LSTM(units = 512,
#                return_sequences = True,
#                )))
# model.add(Dropout(0.4))
# model.add(GRU(units = 256,
#                return_sequences = True
#                ))
# model.add(GRU(units = 256,
#                return_sequences = False
#                ))
# model.add(Dense(256))
# model.add(Dropout(0.4))
# model.add(Dense(trainY.shape[1]))


# model.build(input_shape = trainX.shape)
model.build(input_shape = (None, seq_len, 1))

model.summary()

learning_rate = 0.0005

optimizer = Adam(learning_rate = learning_rate)

model.compile(loss = 'mse', optimizer = optimizer)

try:
    model.load_weights(WEIGHTS_USING)
    print("Loaded model weights from disk")
except Exception as e:
    print(e)
    print("No weights found, training model from scratch")

    history = model.fit(trainX, trainY, epochs=200, batch_size=256,
                    validation_split=0.1, verbose=1)

    model.save_weights(WEIGHTS_USING)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

testX = []
testX.append(train_data_scaled[n_train-seq_len:n_train, COLUMNS_MATCHING[PREDICT_TARGET]])
testX = np.array(testX)

prediction = model.predict(testX)
prediction = prediction.reshape(pred_days, -1)
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
mean_values_pred[:, 0] = np.squeeze(prediction)
y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

plt.plot(y_pred[:],
         color='red',
         linestyle='--',
         label='Predicted Open Price')

plt.show()

# testX, testY = [], []
# testX.append(train_data_scaled[n_train-seq_len:n_train, 1])
# testY.append(test_data_scaled[:pred_days, 1])

# testX, testY = np.array(testX), np.array(testY)

# prediction = model.predict(testX)

# prediction = prediction.reshape(pred_days, -1)
# testY = testY.reshape(pred_days, -1)

# mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
# mean_values_pred[:, 0] = np.squeeze(prediction)
# y_pred = scaler.inverse_transform(mean_values_pred)[:,0]
# mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

# mean_values_testY[:, 0] = np.squeeze(testY)

# testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

# plt.plot(test_dates[:pred_days],
#          testY_original[:],
#          color='blue',
#          label='Actual Open Price')

# plt.plot(test_dates[:pred_days],
#          y_pred[:],
#          color='red',
#          linestyle='--',
#          label='Predicted Open Price')

# plt.show()



# prediction = model.predict(testX)
# print(prediction.shape, testY.shape)

# print(model.evaluate(testX, testY, verbose=1))

# # generate array filled with means for prediction
# mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# # substitute predictions into the first column
# mean_values_pred[:, 0] = np.squeeze(prediction)

# # inverse transform
# y_pred = scaler.inverse_transform(mean_values_pred)[:,0]
# print(y_pred.shape)

# # generate array filled with means for testY
# mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

# # substitute testY into the first column
# mean_values_testY[:, 0] = np.squeeze(testY)

# # inverse transform
# testY_original = scaler.inverse_transform(mean_values_testY)[:,0]
# print(testY_original.shape)

# # plotting
# plt.figure(figsize=(14, 5))

# # plot original 'Open' prices
# plt.plot(dates, y, color='green', label=f'Original {PREDICT_TARGET}')

# # plot actual vs predicted
# plt.plot(test_dates[seq_len:], testY_original, color='blue', label='Actual Open Price')
# plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='Predicted Open Price')
# plt.xlabel('Date')
# plt.ylabel('Open Price')
# plt.title('Original, Actual and Predicted Open Price')
# plt.legend()
# plt.show()

# # Calculate the start and end indices for the zoomed plot
# zoom_start = len(test_dates) - 260
# zoom_end = len(test_dates)

# # Create the zoomed plot
# plt.figure(figsize=(14, 5))

# # Adjust the start index for the testY_original and y_pred arrays
# adjusted_start = zoom_start - seq_len

# plt.plot(test_dates[zoom_start:zoom_end],
#          testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
#          color='blue',
#          label='Actual Open Price')

# plt.plot(test_dates[zoom_start:zoom_end],
#          y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
#          color='red',
#          linestyle='--',
#          label='Predicted Open Price')

# plt.xlabel('Date')
# plt.ylabel('Open Price')
# plt.title('Zoomed In Actual vs Predicted Open Price')
# plt.legend()

# plt.show()