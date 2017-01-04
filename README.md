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