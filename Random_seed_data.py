import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib import RC

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.rcParams["figure.figsize"] = (16, 10)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

data_time = np.arange(0, 200, 0.1)
sin_values = np.sin(data_time) + np.random.normal(scale=0.5, size=len(data_time))
#plt.plot(data_time, sin_values, label='sine (with noise)')

data_full = pd.DataFrame(dict(sine=sin_values), index=data_time, columns=['sine'])
#print(data_full.head())

len_train = int(len(data_full) * 0.8)
len_test = len(data_full) - len_train
train, test = data_full.iloc[0:len_train], data_full.iloc[len_train:len(data_full)]

def gen_data(X, y, num_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - num_steps):
        Xs.append(X.iloc[i:(i + num_steps)].values)       
        ys.append(y.iloc[i + num_steps])
    return np.array(Xs), np.array(ys)
    
num_steps = 10
trainX, trainY = gen_data(train, train.sine, num_steps)
testX, testY = gen_data(test, test.sine, num_steps)
