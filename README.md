# Predict River Total Phosphorous: SMOGN plus Machine-Learning algorithms (Support Vector Regressor and XGboost)
An integrated approach based on the correction of imbalanced small datasets and the application of machine learning algorithms to predict Total Phosphorus concentration in rivers.


This repository includes the python code of two models that were used to predict the total phosphorous concentration of 22 rivers with limiting forcing data. The input datasets are divided between a training (80% of the entire dataset) and a testing dataset (the remaining 20%). Then 100 different training datasets are derived for each station from the initial dataset through the application of the Synthetic Minority Over-Sampling Technique for regression with Gaussian Noise (SMOGN) (Branco et al. 2017). The five SMOGN parameters that drive the algorithm are randomly derived at each model run considering a pre-defined parameter space. After, this initial assessment, two models are fitted to the training datasets: Extreme Gradient Boosting (XGboost) and Support Vector Regressor (SVR). The models hyperparameters are optimized with a Bayesian optimization algorithm (Hyperopt) 


The results of this study are described in the following manuscript: 
**Almeida, M.C. and Coelho P.S.: An integrated approach based on the correction of imbalanced small datasets and the application of machine learning algorithms to predict Total Phosphorus concentration in rivers**:

- XGboost (_vide_ [XGboost webpage](https://xgboost.readthedocs.io/en/stable/) (Chen and Guestrin, 2016))
-	SVR (_vide_ [sklearn webpage](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html))
-	SMOGN (_vide_ [SMOGN webpage](https://github.com/nickkunz/smogn))

The machine learning models hyperparameter optimization was implemented with the Tree-structured Parzen Estimators algorithm (TPE) (Bergstra et al 2011). The python code implementation of TPE with the Hyperot algorithm (Bergstra et al 2013) is also available.


## Input data

In the folder Data we have included 22 input files. These files include the following five columns:

1. DATE (MMDDYEAR);
2. Observed Total Phosphorus concentration,(mg/l);
3. Discharge,(m<sup>3</sup>s<sup>-1</sup>);
4. Month of the year (e.g. 1, 2, 3,..., 12);
5. Mean daily water temperature,(°C).

## Hyperparameter optimization
It is easy to find the model parameters in the code. Nonetheless, in the following table we have included the models parameters that are optimized with the TPE algorithm.

#### Table1. Model parameters and optimization range
Model|	Prior distribution|	Parameter     |	Optimization range
---- | ------------------ | ------------- | ------------------ 
XGboost         |Categorical        |	'boster'|	['gbtree', 'gblinear', 'dart'] 
XGboost         |uniform            |	'colsample_bylevel' |	[0, 1] 
XGboost         |uniform            |	'colsample_bytree'|[0, 1] 
XGboost         |uniform            |	'gamma'|[0, 10] 
XGboost         |uniform            |	'learning_rate' |	[0, 1]
XGboost         |uniform            |	'n_estimators'| [1, 300] 
XGboost         |uniform            |	'subsample' |  [0, 1]
SVR             |Categorical        |	'C'|[0.1,1,100,1000]
SVR             |Categorical        |	'kernel'|	['rbf','poly','sigmoid','linear']
SVR         	  |Categorical        |'degree' |	[1,2,3,4,5,6]
SVR             |Categorical        |	'gamma' |	[1, 0.1, 0.01, 0.001, 0.0001]
SVR             |Categorical        |	'epsilon'|	[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]


## How to run the models
1. Install XG boost, SVR and hyperopt from the repositories
2. Create an empty folder;
3. In this folder place the python code file (e.g. Hyper_ANN.py) and the input file (e.g. st1.xlsx); In the code file (e.g. Hyper_ANN.py) set the training and validation percentages of the dataset (e.g. train_size=0.7, test_size=0.3);
4. Run the code. The output includes: file with score for each model run; file with the parameters for each model run; file with the Mean Average Error (MAE) for the training dataset; file with the MAE for the validation dataset. 

## How to run the optimized models
5. Create an empty folder;
6. In this folder place the python code file (e.g. XG_BOOST_SMOGN.py) and the input file or files (e.g. st1.xlsx; st2.xlsx; st3.xlsx;...;st22.xlsx).
7. Run the code. The output includes: 
- files with the modified initial dataset obtained with SMOGN for each iteration (e.g. SMOGN_out1.xlsx);
- file with the SMOGN parameters obtained for each iteration (e.g. SMOGN_parameters1.xlsx)
- file with the machine learning model hyperparameters, one file per station (e.g.parameters1.csv); 
- file with the metrics MAE and NSE for the machine learning model and for each station obtained for each hyperopt iteration (e.g. metrics1.xlsx)
- file with the predicted values for the training dataset for each station (e.g.trainSVR1.xlsx); 
- file with the predicted values for the testing dataset for each station (e.g.testSVR1.xlsx.xlsx); 
- file with the total phosphorus prediction considering the entire dataset (e.g.prediction.xlsx)



## References
Almeida, M.C. and Coelho P.S.: An integrated approach based on the correction of imbalanced small datasets and the application of machine learning algorithms to predict Total Phosphorus concentration in rivers,...

Bergstra, J. S., Bardenet, R., Bengio, Y. and Kegl, B.: Algorithms for hyper-parameter optimization, in Advances in Neural Information Processing Systems, 2011, 2546–2554, 2011.

Bergstra, J., Yamins, D., Cox, D. D.: Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), 115-23, 2013.

Chen, T., Guestrin, C., 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939785.

Branco, P., Ribeiro, R. P., Torgo, L., Krawczyk, B., Moniz, N., 2017. Smogn: a pre-processing approach for imbalanced regression, Proceedings of Machine Learning Research 74, 36–50.
