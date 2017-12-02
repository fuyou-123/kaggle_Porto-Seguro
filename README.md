# kaggle(Porto Seguro competition)
This is a binary classification project. The dataset is easy to analyse compared with other kaggle competitions. I learned a lot about parameter tuning in this project and also learned how to use AWS. 
## Features
* Binary feature
* Categorical feature
* Continuous feature
* The dataset has missing value for some features
## Models
* xgboost
* LibFM(pywFM for Python)
* LightGBM
* randomforest
## Methods
One hot encoding for categorical features is the common way. Cross validation is used to do parameter tuning and prevent overfitting. Stacking is used to improve prediction accuracy.
## Additional
Though the dataset is small, it is computionally expensive using one hot encoding especially for xgboost and libFM model. I used Amazon Web Service to run Jupyter code. 
