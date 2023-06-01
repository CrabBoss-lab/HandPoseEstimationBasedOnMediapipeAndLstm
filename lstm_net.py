# -*- codeing = utf-8 -*-
# @Time :2023/4/13 9:45
# @Author :yujunyu
# @Site :
# @File :00_model.py
# @software: PyCharm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

actions = np.array(['normal', 'r_hand', 'a_hand'])

# 搭建模型
model = Sequential()
# LSTM中的relu改为tanh，提升了cpu的推理速度
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.summary()
