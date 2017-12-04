import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score


header = ['Id', 'SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
df = pd.read_csv('Iris.csv', names=header,delimiter=',')

train_data, test_data = cv.train_test_split(df, test_size=0.33)

dici = {'Iris-versicolor':1,'Iris-virginica':2,'Iris-setosa':3}
dici2 = {1:'Iris-versicolor', 2:'Iris-virginica', 3:'Iris-setosa'}

matriz_x = []
matriz_y = []

for line in train_data.itertuples():
    linha = []
    linha.append(float(line[2]))
    linha.append(float(line[3]))
    linha.append(float(line[4]))
    linha.append(float(line[5]))
    matriz_x.append(linha)
    matriz_y.append(dici[line[6]])

algo = GaussianNB()

algo.fit(np.array(matriz_x),np.array(matriz_y))

acertos = 0
erros = 0

for line in test_data.itertuples():
    linha = []
    linha.append(float(line[2]))
    linha.append(float(line[3]))
    linha.append(float(line[4]))
    linha.append(float(line[5]))
    correto = line[6]
    resposta = algo.predict_proba(np.array([linha]))
    predict_int = np.array(resposta, dtype=np.int)
    print(predict_int)
    # if dici2[resposta[0]] == correto:
    #     acertos += 1
    # else:
    #     erros += 1
    # print('resposta:', dici2[resposta[0]], 'correto:', correto)
    print('resposta:', resposta, 'correto:', correto)
print('acertos:', acertos, 'erros:', erros)