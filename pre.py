import OpenDartReader
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout,Dense, Activation
import datetime

api_key = 'f895aea46ddc13d5ffc0e3d38dcbd1058ca64462'
dart = OpenDartReader(api_key)
com = str(input("보고싶은 주가그래프의 주식이름을 적으세요")) #회사명


href1="https://query1.finance.yahoo.com/v7/finance/download/"#005930.KS


href="https://query1.finance.yahoo.com/v7/finance/download/"

href2="?period1=1484265600&period2=1605225600&interval=1d&events=history&includeAdjustedClose=true"

print(code_)


data = pd.read_csv()#주가 csv파일

high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices)/2
print(high_prices, low_prices, mid_prices)

seq_len = 50 # window 값이 50
sequence_length = seq_len + 1

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])

#50개를 보고 한개를 예측하고 윈도우에 들어가는 총 데이터는 50 + 1 갯수가 된다.
#데이터의 정규화
#모델을 잘 예측을 하기 위해
normalized_data = []

for window in result:
    normalized_window = [[(float(p) / float(window[0]))-1]for p in window]  #
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# 트레이닝셋과 테스트셋 9:1비율로 나누 

row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)
#데이터 셔플
x_train = train[:,:-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]



x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[row:, -1]

model = Sequential()


model.add(LSTM(50, return_sequences = True, input_shape=(50,1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()
#손실함수 mse
#옵티마이저 ="rmsprop"
#결과 값은 50개가 들어가서 한 개가 나온다.
#LSTM 의 n의 값은 유닛 수를 의미하고 유닛 수를 조정하면서 성능을 측정을 할 수 있다.
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=40, epochs=20)


#그래프를 그려서 예측을 하는 것.
#model.predict 를 사용하여 모델을 예측하는 것.
#
#x_test 데이터를 예측을 하고 pred라는 변수에 담는다.
pred = model.predict(x_test)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
