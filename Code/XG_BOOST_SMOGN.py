import numpy as np
import pandas as pd
import os
import glob
import smogn
import random
import logging
import hyperopt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
from hyperopt import hp
from hyperopt import STATUS_OK
from pprint import pprint
from random import randrange


#===============================================================================
# Define path to input files
#===============================================================================

path = os.getcwd()
xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))

#===============================================================================
# Metric
#===============================================================================
def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

#===============================================================================
# Main loop
#===============================================================================
k_out=[]
samp_method_out=[]
rel_thres_out=[]
rel_xtrm_type_out=[]
rel_coef_out=[]
mae_out=[]
nse_out=[]

for f in xlsx_files:
    for i in range(100):
        # Import data
        data = pd.read_excel(f, index_col=0, header=0)
        data.columns = ['Ptotal','Discharge','Month','WaterT']
        
#===============================================================================
# Define data
#===============================================================================
        X = data.iloc[:, 1:]
        y = data.loc[:, ['Ptotal']]
        
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=False, stratify=None)
        data_train = pd.DataFrame(np.column_stack([XTrain, yTrain]), columns=['Discharge','Month', 'WaterT', 'Ptotal'])

#===============================================================================
# SMOGN Parameter search space
#===============================================================================
        k=randrange(1, 10)
        d = {'extreme':'extreme', 'balance':'balance'}
        samp_method = random.choice(list(d.values()))
        rel_thres =random.uniform(0, 1) #real number between 0 and 1
        
        c = {'high':'high', 'both':'both'}
        rel_xtrm_type = random.choice(list(c.values()))
        rel_coef = random.uniform(0.01, 0.4)
        
        
        k_out.append(k)
        samp_method_out.append(samp_method)
        rel_thres_out.append(rel_thres)
        rel_xtrm_type_out.append(rel_xtrm_type)
        rel_coef_out.append(rel_coef)

        
        try:
            data_smogn = smogn.smoter(
            data = data_train, 
            y = 'Ptotal', 
            k = k, 
            samp_method = samp_method,
            rel_thres = rel_thres, #It specifies the threshold of rarity. The higher the threshold, the higher the over/under-sampling boundary. The inverse is also true, where the lower the threshold, the lower the over/under-sampling boundary. 
            rel_method = 'auto',
            rel_xtrm_type = rel_xtrm_type,
            rel_coef = rel_coef) #It specifies the box plot coefficient used to automatically determine extreme and therefore rare "minority" values in y
            X = data_smogn.drop(columns=['Ptotal'])
            y = data_smogn[['Ptotal']]
            smogn_out=pd.concat([X,y],axis=1)
            
#===============================================================================
# Export SMOGN results to Excel
#===============================================================================
        
            writer = pd.ExcelWriter(os.path.basename(f)+ 'SMOGN_out'+str(i)+'.xlsx')
            smogn_out.to_excel(writer,'train')
            writer.save()
#===============================================================================
# Define train and test datasets
#===============================================================================            
            
            XTrain = smogn_out.loc[:, smogn_out.columns != 'Ptotal']
            yTrain = smogn_out.loc[:, ['Ptotal']].values.ravel()
            XTest = XTest
            yTest = yTest.values.ravel()
            
#===============================================================================
# Retrieve best hyperopt parameters
#===============================================================================
            
            def getBestModelfromTrials(trials):
                valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
                losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
                index_having_minumum_loss = np.argmin(losses)
                best_trial_obj = valid_trial_list[index_having_minumum_loss]
                return best_trial_obj['result']['Trained_Model']
            
#===============================================================================
# Define parameters in hyperopt
#===============================================================================
            
            def uniform_int(name, lower, upper):
                # `quniform` returns:
                # round(uniform(low, high) / q) * q
                return hp.quniform(name, lower, upper, q=1)
            
            def loguniform_int(name, lower, upper):
                # Do not forget to make a logarithm for the
                # lower and upper bounds.
                return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)
            
            parameter_space = {
           
               'subsample':hp.uniform('subsample', 0, 1),
               
               'colsample_bytree':hp.uniform('colsample_bytree', 0, 1),
               
               'colsample_bylevel':hp.uniform('colsample_bylevel', 0, 1),
               
               'gamma':hp.uniform('gamma', 0, 10),
               
               'boster':hp.choice('boster', ['gbtree','gblinear','dart']),
                      
               'learning_rate': hp.uniform('learning_rate', 0, 1),
                      
                'n_estimators': hp.uniform('n_estimators', 1, 300),
            }
            
#===============================================================================
# Construct a function that we want to minimize
#===============================================================================
            parameters_out=[]
            maetrain_out=[]
            maeTest_out=[]
            
            def train_network(parameters):
                parameters_out.append(parameters)
                print("Parameters:")
                pprint(parameters)
                print()    
                
                subsample=parameters['subsample']
                colsample_bytree=parameters['colsample_bytree']
                colsample_bylevel=parameters['colsample_bylevel']
                gamma=parameters['gamma']
                boster=parameters['boster']
                learning_rate=parameters['learning_rate']
                n_estimators=int(parameters['n_estimators'])
                
                XScaler = StandardScaler()
                XScaler.fit(XTrain)
                XTrainScaled = XScaler.transform(XTrain)
                XTestScaled = XScaler.transform(XTest)
            
                yScaler = StandardScaler()
                yScaler.fit(yTrain.reshape(-1, 1))
                yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
                yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
            
                #regressor
                regressor = xgb.XGBRegressor(subsample=subsample,colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,gamma=gamma,boster=boster,learning_rate=learning_rate, n_estimators=n_estimators) 
                regressor.fit(XTrainScaled,yTrainScaled) 
            
            
#===============================================================================
#  Define metrics
#===============================================================================
                yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
                yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
                maetrain = np.mean(np.abs(yTrain-yEstTrain))
                maeTest = np.mean(np.abs(yTest-yEstTest))
                maetrain_out.append(maetrain)
                maeTest_out.append(maeTest)
                
                return {'loss':maeTest, 'status': STATUS_OK, 'Trained_Model':regressor}
#===============================================================================
# run hyperparameter optimization
#===============================================================================
            # Object stores all information about each trial.
            # Also, it stores information about the best trial.
            trials = hyperopt.Trials()
            
            tpe = partial(
                hyperopt.tpe.suggest,
            
                # Sample 1000 candidate and select candidate that
                # has highest Expected Improvement (EI)
                n_EI_candidates=10,
            
                # Use 20% of best observations to estimate next
                # set of parameters
                gamma=0.2,
            
                # First 20 trials are going to be random
                n_startup_jobs=10,
            )
            
            hyperopt.fmin(
                train_network,
            
                trials=trials,
                space=parameter_space,
            
                # Set up TPE for hyperparameter optimization
                algo=tpe,
            
                # Maximum number of iterations. Basically it trains at
                # most 200 networks before selecting the best one.
                max_evals=100,
            )
            
                       
            # Output parameters file for all iterations
            df2 = pd.DataFrame (parameters_out)
            df2.to_csv(os.path.basename(f)+'parameters'+str(i)+'.csv')

#===============================================================================
# get best model from trials
#===============================================================================                             
 
            model = getBestModelfromTrials(trials)
            parameters=model.get_params()
            subsample=parameters['subsample']
            colsample_bytree=parameters['colsample_bytree']
            colsample_bylevel=parameters['colsample_bylevel']
            gamma=parameters['gamma']
            boster=parameters['boster']
            learning_rate=parameters['learning_rate']
            n_estimators=int(parameters['n_estimators'])
            
#===============================================================================
# run SVR with best hyperparameters
#===============================================================================
            
            
            XScaler = StandardScaler()
            XScaler.fit(XTrain)
            XTrainScaled = XScaler.transform(XTrain)
            XTestScaled = XScaler.transform(XTest)
            
            yScaler = StandardScaler()
            yScaler.fit(yTrain.reshape(-1, 1))
            yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
            yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
            
            regressor = xgb.XGBRegressor(subsample=subsample,colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bytree,gamma=gamma,boster=boster,learning_rate=learning_rate, n_estimators=n_estimators)
            regressor.fit(XTrainScaled,yTrainScaled)
            
            #Actual value and the predicted value
            yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
            yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
            
            results_train = pd.DataFrame({'obs': yTrain, 'est': yEstTrain}, index = XTrain.index)
            results_test = pd.DataFrame({'obs': yTest, 'est': yEstTest }, index = XTest.index)
            
               
            #Export results to Excel
            writer = pd.ExcelWriter(os.path.basename(f)+ 'trainSVR'+str(i)+'.xlsx')
            results_train.to_excel(writer,'train')
            writer.save()
            
            writer = pd.ExcelWriter(os.path.basename(f)+ 'testSVR'+str(i)+'.xlsx')
            results_test.to_excel(writer,'test')
            writer.save()
            
            #coefficients.to_excel(writer,'coefficients')
            
            # Metrics
            mae = np.mean(np.abs(yTrain-yEstTrain))
            mse  = np.mean((yTrain-yEstTrain)**2)
            mseTes = np.mean((yTest-yEstTest)**2)
            maeTes = np.mean(np.abs(yTest-yEstTest))
            meantrain = np.mean(yTrain)
            ssTest = (yTrain-meantrain)**2
            r2=(1-(mse/(np.mean(ssTest))))
            meantest = np.mean(yTest)
            ssTrain = (yTest-meantest)**2
            r2Tes=(1-(mseTes/(np.mean(ssTrain))))
            nse=nashsutcliffe(yTest, yEstTest)
            
            mae_out.append(mae)
            nse_out.append(nse)
            
            # Plot results
            print("NN MAE: %f (All), %f (Test) " % (mae, maeTes))
            print ("NN MSE: %f (All), %f (Test) " % (mse, mseTes))
            print ("NN R2: %f (All), %f (Test) " % (r2, r2Tes))
                
                
#===============================================================================
# Predict with unseen data
#===============================================================================
            
            dataR = pd.read_excel(f, index_col=0, header=0)
            dataR.columns = ['Ptotal','Discharge','Month', 'WaterT']
            
            
            #Define data
            XR = dataR.iloc[:, 1:] # X = data.iloc[:, 1:2]
            yR = dataR.loc[:, ['Ptotal']]
            
            
            XTestR = XR
            yTestR = yR.values.ravel()
            XTestScaledR = XScaler.transform(XTestR)
            yTestScaledR = yScaler.transform(yTestR.reshape(-1, 1)).ravel()
            yTestPredictedR = yScaler.inverse_transform(regressor.predict(XTestScaledR).reshape(-1, 1)).ravel()
            
            results2 = pd.DataFrame({'obs': yTestR, 'est': yTestPredictedR}, index = XTestR.index)
            
            
           
            #Export results to Excel
            writer = pd.ExcelWriter(os.path.basename(f)+'prediction'+'.xlsx')
            results2.to_excel(writer,'results')
            writer.save()
         
        except:
              pass
    
    #Export results to Excel
    model_out = pd.DataFrame( {'mae_out': mae_out, 'nse_out': nse_out})
    writer = pd.ExcelWriter('A -'+os.path.basename(f)+'model_out'+str(i)+'.xlsx')
    model_out.to_excel(writer,'mae_all')
    writer.save()   
    SMOGN_parameters_out = pd.DataFrame( {'k': k_out,'samp_method': samp_method_out, 'rel_thres': rel_thres_out,'rel_xtrm_type': rel_xtrm_type_out, 'rel_coef': rel_coef_out})
    writer = pd.ExcelWriter(os.path.basename(f)+'SMOGN_parameters_out'+str(i)+'.xlsx')
    SMOGN_parameters_out.to_excel(writer,'results')
    writer.save()
