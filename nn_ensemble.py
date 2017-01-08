import numpy as np
import pandas as pd

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l1
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as mpatches
import seaborn as sn

np.random.seed(42)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = np.log1p(train.SalePrice)
outliers_id = np.array([524, 1299])
y_train = y_train.drop(outliers_id)


# Function to load in all models.
def load_model(model_name, n_folds=10):
    ''' Input the model name to be loaded, and n_folds used.
    Returns the model that is aggregated from weights predicted from CV sets. '''

    train = []
    for i in range(n_folds):
        train.append(np.loadtxt('model_4/{}_pred_fold_{}.txt'.format(model_name, i)))

    return train


# Function to load in all test set.
def load_test(model_name):
    ''' Input the model name to be loaded, and n_folds used.
    Returns the model that is aggregated from weights predicted from CV sets. '''

    test = []
    test.append(np.loadtxt('model_4_test_set/{}_test_pred.txt'.format(model_name)))

    return test

### Stacker model (Lasso + Ridge + XGB + KNN) using NN.
### Aggregate weights to be passed into layer 2 model
# 1. This is Lasso predicted weights from Kfold training set
train_lasso = load_model('ls', 10)
train_lasso_folds = np.hstack((train_lasso[0], train_lasso[1], train_lasso[2], train_lasso[3], \
                               train_lasso[4], train_lasso[5], train_lasso[6], \
                               train_lasso[7], train_lasso[8], train_lasso[9]))
print "\nChecking Lasso trainin set size: ", train_lasso_folds.shape

rmse_check_1 = np.sqrt(mean_squared_error(np.log(train_lasso_folds), y_train))
print "Lasso RMSE: ", rmse_check_1

lasso_pred = np.array(load_test('lasso')).T
print lasso_pred.shape


# 2. This is Ridge predicted weights from Kfold training set
train_ridge = load_model('ridge', 10)
train_ridge_folds = np.hstack((train_ridge[0], train_ridge[1], train_ridge[2], train_ridge[3], \
                               train_ridge[4], train_ridge[5], train_ridge[6], \
                               train_ridge[7], train_ridge[8], train_ridge[9]))
print "\nChecking Ridge trainin set size: ", train_ridge_folds.shape

rmse_check_2 = np.sqrt(mean_squared_error(np.log(train_ridge_folds), y_train))
print "Ridge RMSE: ", rmse_check_2

ridge_pred = np.array(load_test('ridge')).T
print ridge_pred.shape

# 3. This is xgb predicted weights from Kfold training set
train_xgb = load_model('xgb', 10)
train_xgb_folds = np.hstack((train_xgb[0], train_xgb[1], train_xgb[2], train_xgb[3], \
                             train_xgb[4], train_xgb[5], train_xgb[6], \
                             train_xgb[7], train_xgb[8], train_xgb[9]))
print "\nChecking XGB training set size: ", train_xgb_folds.shape

rmse_check_3 = np.sqrt(mean_squared_error(np.log(train_xgb_folds), y_train))
print "XGB RMSE: ", rmse_check_3

xgb_pred = np.array(load_test('xgb')).T
print xgb_pred.shape


# 4. This is knn predicted weights from Kfold training set
train_knn = load_model('knn', 10)
train_knn_folds = np.hstack((train_knn[0], train_knn[1], train_knn[2], train_knn[3], \
                             train_knn[4], train_knn[5], train_knn[6], \
                             train_knn[7], train_knn[8], train_knn[9]))
print "\nChecking knn training set size: ", train_knn_folds.shape

rmse_check_4 = np.sqrt(mean_squared_error(np.log(train_knn_folds), y_train))
print "knn RMSE: ", rmse_check_4

knn_pred = np.array(load_test('knn')).T
print knn_pred.shape

# 5. This is rf predicted weights from Kfold training set
train_rf = load_model('rf', 10)
train_rf_folds = np.hstack((train_rf[0], train_rf[1], train_rf[2], train_rf[3], \
                            train_rf[4], train_rf[5], train_rf[6], \
                            train_rf[7], train_rf[8], train_rf[9]))
print "\nChecking rf training set size: ", train_rf_folds.shape

rmse_check_5 = np.sqrt(mean_squared_error(np.log(train_rf_folds), y_train))
print "rf RMSE: ", rmse_check_5

rf_pred = np.array(load_test('rf')).T
print rf_pred.shape

# Resize the ridge and knn prediction so they can fit into the stacker.
xgb_resized = np.resize(xgb_pred, (1459,))
lasso_resized = np.resize(lasso_pred, (1459,))
ridge_resized = np.resize(ridge_pred, (1459,))
rf_resized = np.resize(rf_pred, (1459,))
knn_resized = np.resize(knn_pred, (1459,))
print '\n', xgb_pred.shape, lasso_pred.shape, ridge_resized.shape, knn_resized.shape, rf_pred.shape

# Stacking starts here.
layer_1_train_x = np.vstack((train_xgb_folds, train_lasso_folds, train_ridge_folds, train_rf_folds)).T
layer_1_test_x = np.vstack((xgb_resized, lasso_resized, ridge_resized, rf_resized)).T
lr = LinearRegression()

def create_model():
    model = Sequential()
    model.add(Dense(1000, input_dim=layer_1_train_x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Activation('linear'))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", optimizer='adam')
    epoch = 200
    model.fit(layer_1_train_x, y_train, nb_epoch=epoch, batch_size=200, validation_split=0.2)
    # hist = fit.history
    # nn_pred = model.predict(x_test)
    return model

model = KerasRegressor(build_fn=create_model, verbose=1)

model_rmse = np.sqrt(-cross_val_score(model, np.log(layer_1_train_x), y_train, cv=5, scoring='neg_mean_squared_error'))
print "\nStacker RMSE: ", (model_rmse)

model_pred = create_model()
final_pred = model_pred.predict(layer_1_test_x)
df_final_pred = pd.DataFrame(np.exp(final_pred), index=test["Id"], columns=["SalePrice"])
print "\n", df_final_pred.head()
df_final_pred.to_csv('submission_nn_4.csv', header=True, index_label='Id') # uncomment if want to submit
