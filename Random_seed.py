import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import Random_seed_data as loaded_data
#from matplotlib import RC

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(128, input_shape=(loaded_data.trainX.shape[1], loaded_data.trainX.shape[2])))
lstm_model.add(keras.layers.Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]

history = lstm_model.fit(
    loaded_data.trainX, loaded_data.trainY, 
    epochs=30, 
    batch_size=16, 
    validation_split=0.1,
    shuffle=False,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)],
)

lstm_model.evaluate(loaded_data.testX)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.show()
#################
lstm_model.save("LSTM_model.h5")
#################

# y_pred = lstm_model.predict(testX)
# plt.plot(testY, marker='.', label="true")
# plt.plot(y_pred, 'r', label="prediction")
# plt.ylabel('Value')
# plt.xlabel('Time Step')
# plt.legend()
# plt.show()

