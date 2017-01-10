import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import manifold
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X1 = train.values[:,1:-1]
X2 = test.values[:,1:]

X = np.concatenate((X1,X2),axis=0)
y1 = train.values[:,-1]
y1 = np.asarray([np.log(y) for y in y1])
#-----------------------------------------
# Preprocessing
#-----------------------------------------
# 1: separate different variable types
cat_idx = []
float_idx = []
int_idx = []
for c in range(X1.shape[1]):
    if type(X1[0,c]).__name__ == 'str':
        cat_idx.append(c)
    if type(X1[0,c]).__name__ == 'float' and X1[1,c]==X1[1,c]:
        float_idx.append(c)
    if type(X1[0,c]).__name__ == 'int':
        int_idx.append(c)
# 2: encode string values into numbers
for c in cat_idx:
    uniques = list(set(X[:,c]))
    tmp_dict = dict(zip(uniques,range(len(uniques))))
    n_enc = np.array([tmp_dict[s] for s in X[:,c]])
    X[:,c] = n_enc
        
# 3: what does an embedding of all int values look like?
print('embedding int values...')
plt.figure(1)
X_int = X[:,np.array(int_idx)]
X_int = np.float64(X_int)
# replace nan
X_int[X_int!=X_int] = 0
X_int-=np.min(X_int,axis=0)
X_int/=(.001+np.max(X_int,axis=0))
tsne = manifold.TSNE(n_components=2,init='pca')
Y_int = tsne.fit_transform(X_int)
#y1-=np.nanmin(y1)
#y1/=np.nanmax(y1)
plt.scatter(Y_int[len(X1):,0],Y_int[len(X1):,1],marker='.',label='test')
sp = plt.scatter(Y_int[:len(X1),0],Y_int[:len(X1),1],c=y1,label='train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of int variables')
plt.savefig('t-SNE_int.png')

# 4: what does an embedding of all string values look like?
print('embedding string values...')
plt.figure(2)
X_str = X[:,np.array(cat_idx)]
# replace nan
X_str[X_str!=X_str] = 0

def onehot(x):
    nx=np.zeros((len(x),max(x)+1))
    for k in range(len(x)):
        nx[k,x[k]] = 1
    return nx

X_tmp = []
for c in range(X_str.shape[1]):
    X_tmp.extend(onehot(X_str[:,c]).T)
X_str = np.asarray(X_tmp).T
tsne = manifold.TSNE(n_components=2,init='pca')
Y_str = tsne.fit_transform(X_str)
#y1-=np.nanmin(y1)
#y1/=np.nanmax(y1)
plt.scatter(Y_str[len(X1):,0],Y_str[len(X1):,1],marker='.',label='test')
sp = plt.scatter(Y_str[:len(X1),0],Y_str[:len(X1),1],c=y1,label='train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of string variables')
plt.savefig('t-SNE_string.png')

# 4: what does an embedding of all int and string values look like?
print('embedding int and string values...')
plt.figure(3)
X_strint = np.concatenate((X_int,X_str),axis=1)
tsne = manifold.TSNE(n_components=2,init='pca')
Y_strint = tsne.fit_transform(X_strint)
plt.scatter(Y_strint[len(X1):,0],Y_strint[len(X1):,1],marker='.',label='test')
sp = plt.scatter(Y_strint[:len(X1),0],Y_strint[:len(X1),1],c=y1,label='train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of int and string variables')
plt.savefig('t-SNE_intstring.png')

# center data at 0 scaled from -0.5 to +0.5 for neural networks
# -> start within the linear region of tanh activation function
X_strint -= .5
X_strint_train = X_strint[:len(X1),:]
X_strint_test = X_strint[len(X1):,:]


from keras.models import Model
from keras.layers import Input,Dense,Dropout
from keras import regularizers
from sklearn.model_selection import KFold

#--------------------------------------
# Deep Neural Network
#--------------------------------------
inp = Input(shape=(X_strint.shape[1],))
D1 = Dropout(.1)(inp)
L1 = Dense(64, init='uniform', activation='tanh')(D1)
D2 = Dropout(.2)(L1)
L2 = Dense(64, init='uniform', activation='tanh')(D2)
D3 = Dropout(.2)(L2)
L3 = Dense(36, init='uniform', activation='tanh')(D3)
D4 = Dropout(.2)(L3)
L4 = Dense(1, init='uniform', activation='tanh')(D4)
# model that is trained to predict prices
model1 = Model(inp,L4)
# models for reading out activations in hidden layers
enc_l1 = Model(inp,L1)
enc_l2 = Model(inp,L2)
enc_l3 = Model(inp,L3)
# compile models
model1.compile(loss='mse', optimizer='adam', metrics=['mse'])
enc_l1.compile(loss='mse', optimizer='adam', metrics=['mse'])
enc_l2.compile(loss='mse', optimizer='adam', metrics=['mse'])
enc_l3.compile(loss='mse', optimizer='adam', metrics=['mse'])

# train
min_y1 = np.min(y1)
max_y1 = np.max(y1)
scaled_y1 = y1 - min_y1
scaled_y1 /= max_y1
# model1.fit(X_strint_train, scaled_y1, nb_epoch=55, batch_size=3, shuffle=True,verbose=2, validation_split=0.25)

# CV
folds = KFold(n_splits=10)
for k, (train_index, validation_index) in enumerate(folds.split(X_strint_train)):
    x_cv_train, x_cv_val = X_strint_train[train_index], X_strint_train[validation_index]
    y_cv_train, y_cv_val = scaled_y1[train_index], scaled_y1[validation_index]

    model1.fit(x_cv_train, y_cv_train, nb_epoch=55, batch_size=3, shuffle=True,verbose=2, validation_split=0.25)
    pred = model1.predict(x_cv_val)
    pred *= max_y1
    pred += min_y1
    np.savetxt('nn_pred_fold_{}.txt'.format(k), np.exp(pred))
    np.savetxt('nn_test_fold_{}.txt'.format(k), y_cv_val)

# predict
# final_pred = model1.predict(X_strint_test)
# final_pred *= max_y1
# final_pred += min_y1
# df_final_pred = pd.DataFrame(np.exp(final_pred), index=test["Id"], columns=["SalePrice"])
# print "\n", df_final_pred.head()
# df_final_pred.to_csv('submission_nn_6.csv', header=True, index_label='Id') # uncomment if want to submit

# get hidden layer activations
P1 = enc_l1.predict(X_strint)
P2 = enc_l2.predict(X_strint)
P3 = enc_l3.predict(X_strint)
# get 2d embeddings
tsne = manifold.TSNE(n_components=2,init='pca')
P1_tsne = tsne.fit_transform(P1)
P2_tsne = tsne.fit_transform(P2)
P3_tsne = tsne.fit_transform(P3)

P1_tsne_train = P1_tsne[:len(X1),:]
P2_tsne_train = P2_tsne[:len(X1),:]
P3_tsne_train = P3_tsne[:len(X1),:]
P1_tsne_test = P1_tsne[len(X1):,:]
P2_tsne_test = P2_tsne[len(X1):,:]
P3_tsne_test = P3_tsne[len(X1):,:]

plt.figure(3)
plt.scatter(P1_tsne_test[:,0],P1_tsne_test[:,1],marker='.',label='test')
sp1 = plt.scatter(P1_tsne_train[:-50,0],P1_tsne_train[:-50,1],c=y1[:-50],label='train')
plt.scatter(P1_tsne_train[-50:,0],P1_tsne_train[-50:,1],marker='^',s=55, c=y1[-50:],label='validation')
plt.colorbar(sp1)
plt.legend(prop={'size':6})
plt.title('t-SNE embedding: deep network - layer1')
plt.savefig('t-SNE_deep_layer1.png')

plt.figure(4)
plt.scatter(P2_tsne_test[:,0],P2_tsne_test[:,1],marker='.',label='test')
sp1 = plt.scatter(P2_tsne_train[:-50,0],P2_tsne_train[:-50,1],c=y1[:-50],label='train')
plt.scatter(P2_tsne_train[-50:,0],P2_tsne_train[-50:,1],marker='^',s=55, c=y1[-50:],label='validation')
plt.colorbar(sp1)
plt.legend(prop={'size':6})
plt.title('t-SNE embedding: deep network - layer2')
plt.savefig('t-SNE_deep_layer2.png')

plt.figure(5)
plt.scatter(P3_tsne_test[:,0],P3_tsne_test[:,1],marker='.',label='test')
sp1 = plt.scatter(P3_tsne_train[:-50,0],P3_tsne_train[:-50,1],c=y1[:-50],label='train')
plt.scatter(P3_tsne_train[-50:,0],P3_tsne_train[-50:,1],marker='^',s=55, c=y1[-50:],label='validation')
plt.colorbar(sp1)
plt.legend(prop={'size':6})
plt.title('t-SNE embedding: deep network - layer3')
plt.savefig('t-SNE_deep_layer3.png')
