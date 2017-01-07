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
submission9.csv	0.11509	- Same as previous. Diff set of features

submission8.csv	0.11789	- Same as previous. CV=10

submission7.csv	0.11772	- feature engineering + linear (lasso + ridge + xgb + knn + rf)

submission6.csv	0.11763	- feature engineering + linear(xgb + lasso)

submission4.csv	0.12393	- (xgb + lasso + ridge) with linear regression

submission.csv	0.12754	- Simple lasso.

submission3.csv	0.12162	- XGB + Lasso with LinearRegression

submission.csv	0.12754	- XGB + Lasso

submission.csv	0.12754 - XGBoost single model