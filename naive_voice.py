
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score


header = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","label"]
df = pd.read_csv('voice.csv', names=header,delimiter=',')

train_data, test_data = cv.train_test_split(df, test_size=0.33)

dici = {"male":1,"female":2}
dici2 = {1:"male",2:"female"}

matriz_x = []
matriz_y = []

for line in train_data.itertuples():
    linha = []
    linha.append(float(line[1]))
    linha.append(float(line[2]))
    linha.append(float(line[3]))
    linha.append(float(line[4]))
    linha.append(float(line[5]))
    linha.append(float(line[6]))
    linha.append(float(line[7]))
    linha.append(float(line[8]))
    linha.append(float(line[9]))
    linha.append(float(line[10]))
    linha.append(float(line[11]))
    linha.append(float(line[12]))
    linha.append(float(line[13]))
    linha.append(float(line[14]))
    linha.append(float(line[15]))
    linha.append(float(line[16]))
    linha.append(float(line[17]))
    linha.append(float(line[18]))
    linha.append(float(line[19]))
    linha.append(float(line[20]))
    matriz_x.append(linha)
    matriz_y.append(dici[line[21]])

algo = GaussianNB()
algo2 = KNeighborsClassifier()

algo.fit(np.array(matriz_x),np.array(matriz_y))
algo2.fit(np.array(matriz_x),np.array(matriz_y))

acertos = 0
erros = 0
acertos2 = 0
erros2 = 0
lista_resultados = []
lista_resultados2 = []
lista_corretos = []

print('test data intertuples',test_data.itertuples)
for line in test_data.itertuples():
    linha = []
    linha.append(float(line[1]))
    linha.append(float(line[2]))
    linha.append(float(line[3]))
    linha.append(float(line[4]))
    linha.append(float(line[5]))
    linha.append(float(line[6]))
    linha.append(float(line[7]))
    linha.append(float(line[8]))
    linha.append(float(line[9]))
    linha.append(float(line[10]))
    linha.append(float(line[11]))
    linha.append(float(line[12]))
    linha.append(float(line[13]))
    linha.append(float(line[14]))
    linha.append(float(line[15]))
    linha.append(float(line[16]))
    linha.append(float(line[17]))
    linha.append(float(line[18]))
    linha.append(float(line[19]))
    linha.append(float(line[20]))
    correto = line[21]
    lista_corretos.append(dici[line[21]])
    resposta = algo.predict(np.array([linha]))
    resposta2 = algo2.predict(np.array([linha]))
    lista_resultados.append(resposta[0])
    lista_resultados2.append(resposta2[0])
    # predict_int = np.array(resposta, dtype=np.int)
    # print(predict_int)
    if dici2[resposta2[0]] == correto:
        acertos2 += 1
    else:
        erros2 += 1
    if dici2[resposta[0]] == correto:
        acertos += 1
    else:
        erros += 1
    print('resposta:', dici2[resposta[0]], 'correto:', correto)
    print('resposta2:', dici2[resposta2[0]], 'correto:', correto)
    # print('resposta:', resposta, 'correto:', correto)
print('acertos:', acertos, 'erros:', erros)
print('precision:',precision_score(lista_corretos,lista_resultados))
print('acuracy:',accuracy_score(lista_corretos,lista_resultados))
print('recall:',recall_score(lista_corretos,lista_resultados))

print('knn acertos:', acertos2, 'knn erros:', erros2)
print('knn precision:',precision_score(lista_corretos,lista_resultados2))
print('knn acuracy:',accuracy_score(lista_corretos,lista_resultados2))
print('knn recall:',recall_score(lista_corretos,lista_resultados2))