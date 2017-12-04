import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

user_dict = {}
book_dict = {}


header = ['user_id', 'isbn', 'rating']
df = pd.read_csv('novo.csv', names=header,delimiter=';')
contador_user = 0
contador_book = 0
for line in df.itertuples():
    user = line[1]
    book = line[2]
    if user not in user_dict:
        user_dict[user] = contador_user
        contador_user += 1
    if book not in book_dict:
        book_dict[book] = contador_user
        contador_book += 1

n_users = df.user_id.unique().shape[0]
n_items = df.isbn.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

train_data, test_data = cv.train_test_split(df, test_size=0.33)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    user = user_dict[line[1]]
    book = book_dict[line[2]]
    train_data_matrix[user,book] = line[3]+1

# test_data_matrix = np.zeros((n_users, n_items),dtype={'user_id':np.int, 'isbn':('U',17), 'rating':np.int})
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    user = user_dict[line[1]]
    book = book_dict[line[2]]
    test_data_matrix[user,book] = line[3]+1



user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print ('Quão esparsa minha base é ' +  str(sparsity*100) + '%')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):

            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                print(i,j)
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    return pred

user_prediction = predict(train_data_matrix, user_similarity, type='user')


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
#pred = predict_topk(train_data_matrix, user_similarity, kind='user', k=40)
#print ('User-based aaaaCF RMSE: ' + str(rmse(pred, test_data_matrix)))