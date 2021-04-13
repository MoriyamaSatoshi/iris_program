#日本大学理工学部応用情報工学科 森山慧
#iris.csvファイルをニューラルネットワークで学習させるプログラム

import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#ニューラルネットワークの作成
model=MLPClassifier(hidden_layer_sizes=(100,100),activation='logistic',solver='adam',max_iter=1000)
df = pd.read_csv('C:/ゼミナール活動/iris.csv')

col1 = ['gakuhen_nagasa','gakuhen_haba','hanabira_nagasa','hanabira_haba']
x = df[col1]
col2=['syurui']
y = df[col2]

#テストデータと訓練データ分割
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#時間計測
start_time = time.perf_counter()

#学習
model.fit(x_train,y_train.values.ravel())

#時間測定
elapsed_time = time.perf_counter()-start_time
print("実行時間[s]　"+str(elapsed_time))

#モデルの精度表示
score = model.score(x_test,y_test)*100
print("正解率　"+str(score) + "%")

#作成したモデルの実行
new=[[0.22,0.63,0.08,0.04]]
result = (model.predict(new))
print("分類結果"+str(result))
