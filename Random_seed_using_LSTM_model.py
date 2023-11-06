from tensorflow import keras
import matplotlib.pyplot as plt
import Random_seed_data as loaded_data

lstm_model = keras.models.load_model("LSTM_model.h5")
y_pred = lstm_model.predict(loaded_data.testX)
plt.plot(loaded_data.testY, marker='.', label="true")
plt.plot(y_pred, 'r', label="prediction")
plt.ylabel('Value')
plt.xlabel('Time Step')
plt.legend()
plt.show()

