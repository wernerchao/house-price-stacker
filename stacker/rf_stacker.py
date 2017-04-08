# Stacker model (Lasso + Ridge + XGB + KNN) using Linear Regression
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# Function to load in all models.
def load_model(model_name, n_folds=10):
    ''' Input the model name to be loaded, and n_folds used.
    Returns the model that is aggregated from weights predicted from CV sets. '''
    train = []
    for i in range(n_folds):
        train.append(np.loadtxt('../model_4/{}_pred_fold_{}.txt'.format(model_name, i)))
    return train

if __name__ == '__main__':
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    y_train = np.log1p(train.SalePrice)
    outliers_id = np.array([523, 1298])
    y_train = y_train.drop(outliers_id)

    ### Aggregate weights to be passed into layer 1 model
    # 1. This is Lasso predicted weights from Kfold training set
    train_lasso = load_model('ls', 10)
    train_lasso_folds = np.hstack((train_lasso[0], train_lasso[1], train_lasso[2], train_lasso[3], \
                                train_lasso[4], train_lasso[5], train_lasso[6], \
                                train_lasso[7], train_lasso[8], train_lasso[9]))
    print "\nChecking Lasso trainin set size: ", train_lasso_folds.shape

    rmse_check_1 = np.sqrt(mean_squared_error(np.log(train_lasso_folds), y_train))
    print "Lasso RMSE: ", rmse_check_1


    # 2. This is Ridge predicted weights from Kfold training set
    train_ridge = load_model('ridge', 10)
    train_ridge_folds = np.hstack((train_ridge[0], train_ridge[1], train_ridge[2], train_ridge[3], \
                                train_ridge[4], train_ridge[5], train_ridge[6], \
                                train_ridge[7], train_ridge[8], train_ridge[9]))
    print "\nChecking Ridge trainin set size: ", train_ridge_folds.shape

    rmse_check_2 = np.sqrt(mean_squared_error(np.log(train_ridge_folds), y_train))
    print "Ridge RMSE: ", rmse_check_2


    # 3. This is xgb predicted weights from Kfold training set
    train_xgb = load_model('xgb', 10)
    train_xgb_folds = np.hstack((train_xgb[0], train_xgb[1], train_xgb[2], train_xgb[3], \
                                train_xgb[4], train_xgb[5], train_xgb[6], \
                                train_xgb[7], train_xgb[8], train_xgb[9]))
    print "\nChecking XGB training set size: ", train_xgb_folds.shape

    rmse_check_3 = np.sqrt(mean_squared_error(np.log(train_xgb_folds), y_train))
    print "XGB RMSE: ", rmse_check_3


    # 4. This is knn predicted weights from Kfold training set
    train_knn = load_model('knn', 10)
    train_knn_folds = np.hstack((train_knn[0], train_knn[1], train_knn[2], train_knn[3], \
                                train_knn[4], train_knn[5], train_knn[6], \
                                train_knn[7], train_knn[8], train_knn[9]))
    print "\nChecking knn training set size: ", train_knn_folds.shape

    rmse_check_4 = np.sqrt(mean_squared_error(np.log(train_knn_folds), y_train))
    print "knn RMSE: ", rmse_check_4


    # 5. This is rf predicted weights from Kfold training set
    train_rf = load_model('rf', 10)
    train_rf_folds = np.hstack((train_rf[0], train_rf[1], train_rf[2], train_rf[3], \
                                train_rf[4], train_rf[5], train_rf[6], \
                                train_rf[7], train_rf[8], train_rf[9]))
    print "\nChecking rf training set size: ", train_rf_folds.shape

    rmse_check_5 = np.sqrt(mean_squared_error(np.log(train_rf_folds), y_train))
    print "rf RMSE: ", rmse_check_5

    ### Read in the prediction on test set. This will be test_x (test set features)
    xgb_pred = np.loadtxt('../model_4_test_set/xgb_test_pred.txt')
    lasso_pred = np.loadtxt('../model_4_test_set/lasso_test_pred.txt')
    ridge_pred = np.loadtxt('../model_4_test_set/ridge_test_pred.txt')
    rf_pred = np.loadtxt('../model_4_test_set/rf_test_pred.txt')
    knn_pred = np.loadtxt('../model_4_test_set/knn_test_pred.txt')


    # Resize the ridge and knn prediction so they can fit into the stacker.
    print '\n', type(xgb_pred), lasso_pred.shape, ridge_pred.shape, knn_pred.shape, rf_pred.shape
    # Stacking starts here.
    layer_1_train_x = np.vstack((train_xgb_folds, train_lasso_folds, train_ridge_folds, train_rf_folds, train_knn_folds)).T
    layer_1_test_x = np.vstack((xgb_pred, lasso_pred, ridge_pred, rf_pred, knn_pred)).T


    ### Use Random Forest to do the stacking
    rf_stack = RandomForestRegressor()
    rf_stack_rmse = np.sqrt(-cross_val_score(rf_stack, np.log(layer_1_train_x), y_train, cv=5, scoring='neg_mean_squared_error'))
    print "\nRF Stacker CV RMSE: ", (rf_stack_rmse.mean())

    rf_stack.fit(np.log(layer_1_train_x), y_train)
    rf_stack_final_pred = rf_stack.predict(layer_1_test_x)
    df_rf_stack_final_pred = pd.DataFrame(np.exp(rf_stack_final_pred), index=test["Id"], columns=["SalePrice"])
    df_rf_stack_final_pred.to_csv('submission13_rf_stack.csv', header=True, index_label='Id')
    print "\n", df_rf_stack_final_pred.head()


    ### User Linear Regression to do the stacking
    lr = LinearRegression()
    lr_rmse = np.sqrt(-cross_val_score(lr, np.log(layer_1_train_x), y_train, cv=5, scoring='neg_mean_squared_error'))
    print "\nLinear Stacker CV RMSE: ", (lr_rmse.mean())

    lr.fit(np.log(layer_1_train_x), y_train)
    final_pred = lr.predict(layer_1_test_x)
    df_final_pred = pd.DataFrame(np.exp(final_pred), index=test["Id"], columns=["SalePrice"])
    df_final_pred.to_csv('submission12_linear_stack.csv', header=True, index_label='Id')
    print "\n", df_final_pred.head()




##################################################################################################################
# Stacker training features: predicted weights from layer_2 models (xgbost, lasso, ...) Kfold training set
# Stacker training target:   real training set target
# Stacker testing features:  predicted weights from layer_2 models test set
# Stacker testing target:    real testing set target (which is on Kaggle, we don't have it)