# House Price Prediction for Houses in Amesm, Iowa
This repository is for Kaggle competition, 
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

I've borrowed ideas from [Alexandru Papiu](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/comments) 
for feature engineering and linear regression, [MeiChengShih](https://www.kaggle.com/mshih2/house-prices-advanced-regression-techniques/using-xgboost-for-feature-selection/comments) 
for feature selection/engineering. 

Overall I used 5 models: lasso + ridge + xgb + knn + rf, and used linear regression to ensemble them together.
I also used 10 fold CV.
I achieved local CV RMSE: 0.10732,
and public leaderboard test RMSE: 0.11763.

Currently, at rank at top 12%.


Folders:
model: lasso + ridge + xgb + knn [3 fold cv]
model_2: lasso + ridge + xgb + knn + rf [3 fold cv]
model_3: lasso + ridge + xgb + knn + rf [10 fold cv]
model_4: same as model_3 with diff features

Submissions:
submission_nn_1:
model = Sequential()
model.add(Dense(1024, input_dim=x_train.shape[1], W_regularizer=l1(0.001)))
model.add(Activation('tanh'))
model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dense(1))
sgd = SGD(lr=0.0001)
model.compile(loss = "mean_squared_error", optimizer=sgd)
mse_test: 1.31

submission_nn_2:
model = Sequential()
model.add(Dense(1024, input_dim=x_train.shape[1], W_regularizer=l1(0.001)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dense(1))
sgd = SGD(lr=0.00005)
model.compile(loss = "mean_squared_error", optimizer=sgd)
mse_test: 2.14373

submission_nn_3:
model = Sequential()
model.add(Dense(500, input_dim=x_train.shape[1], W_regularizer=l1(0.001)))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('linear'))
model.add(Dense(1))
sgd = SGD(lr=0.00005)
adam = Adam(lr=0.0001)
model.compile(loss = "mean_squared_error", optimizer=adam)
fit = model.fit(x_tr, y_tr, nb_epoch=200, batch_size=300, validation_data = (x_val, y_val))
hist = fit.history
nn_pred = model.predict(x_test)
mse_test: 4.55097

submission_nn_4:
Single model nn (used 1410 data sets)

submission_nn_5:
Single model nn (used all 1460 data sets)

submission9.csv	0.11509	- Same as previous. Diff set of features

submission8.csv	0.11789	- Same as previous. CV=10

submission7.csv	0.11772	- feature engineering + linear (lasso + ridge + xgb + knn + rf)

submission6.csv	0.11763	- feature engineering + linear(xgb + lasso)

submission4.csv	0.12393	- (xgb + lasso + ridge) with linear regression

submission.csv	0.12754	- Simple lasso.

submission3.csv	0.12162	- XGB + Lasso with LinearRegression

submission.csv	0.12754	- XGB + Lasso

submission.csv	0.12754 - XGBoost single model
