from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
naver = pd.read_csv('005930.KS.csv') #삼성 주식번호

high_prices = data['High'].values       #최고가
low_prices = data['Low'].values         #최저가
mid_prices = (high_prices+low_prices)/2 #중가

seq_len = 50                            #window 값이 50 (50일 주가를 보고)
seq_length = seq_len + 1                #+1한것을 추측

result = []
for i in range(len(mide_prices) - seq_length):
    result.append(mid_prcies[i:i_seq_length]) #result에 list 51개를 한스텝씩 저장

normalized_data = []                            #데이터 정규화 모델을 잘예측하기 위한 과정
for window in result:
    normalized_window = [[(float(p) / float(window[0]))-1]for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

x_train = train[:,:-1]
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
y_train= train[:, -1]
x_test = result[:,-1]
x_test = np.result[row:,:-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[row:,-1]


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(50,1)))
model.add(LSTM(64, return_sequences = False,))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse',optimizer='rmsprop')

model.fit(x_train, y_train, validation_data=(x_test, u_test),batch_size=10, epochs=20)

pred = model.predict(x_test)

fig(plt.figure(facecolor='white'))
ax = fig.add_subplot(111)
ax.plot(y_test,label='실주가')
ax.plot(pred,label='예측')
ax.legend()
plt.show()    
