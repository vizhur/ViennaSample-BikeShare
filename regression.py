########################################################################################################################
# This program builds regression model for bike forecasting
# Inputs: expects train and test data files with following
#          columns (in order, the names are immaterial)
#         Hour	Weekday	start station id	RideInitiationCount	N_DryBulbTemp	N_RelativeHumidity	N_WindSpeed
# Outputs: pickled/numpy files for regression model and tranforms
#          also prints the R-Sq on Training data and RMSE for test
########################################################################################################################

#############################################
# Import the necessary modules and libraries
#############################################
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import csv
import os
import os.path
import pickle
print(__doc__)

#############################################
# Input parser for the train/test data files
#############################################
def readdata(filepath, filename, labelcolumn, excludedcatcolumns, excludednumcolumns):
    csvfile = open(os.path.join(filepath,filename))
    reader = csv.reader(csvfile, delimiter=',')

    headers = next(reader)

    nanfloat = float('nan')
    nanint = 0

    parsed = ((row[0],
               row[1],
               row[2],
               nanint if (row[3] == 'NA' or row[3] == 'nan') else int(row[3]),
               nanfloat if row[4] == 'NA' else float(row[4]), 
               nanfloat if row[5] == 'NA' else float(row[5]), 
               nanfloat if row[6] == 'NA' else float(row[6]), 
               )     
              for row in reader)

    selector = [x for x in range(len(headers)) if (x not in excludedcatcolumns and x not in excludednumcolumns)]

    data = []
    label = []

    for row in parsed:
        data.append([row[i] for i in selector])
        label.append(row[labelcolumn])
    return{'data':data,'label':label}


#############################################
# Preprocessing the features using
# Label Encoding for categorical features and
# Missing value Imputation for Numericals
#############################################
def processfeatures(features, categoricalfeaturestart, categoricalfeaturestartend, testdata = None):   

    istest = False
    if not testdata:
        istest = True

    numfeatures = len(features[0])
    processedfeatures = np.empty([len(features),numfeatures])
    featureencoders = np.empty((categoricalfeaturestartend,), dtype=np.object)
    
    #loading the categorical to numerical encoders if running on test data
    if istest:
        for iter in range(0, categoricalfeaturestartend):
            featureencoders[iter] = preprocessing.LabelEncoder()
            featureencoders[iter].fit(np.array(np.load('./outputs/'+str(iter)+'_labelencoder.npy')))
    
    #categorical to numerical transform
    for iter in range(0, numfeatures):
        if(iter >= categoricalfeaturestart and iter < categoricalfeaturestartend):   
            encoder = preprocessing.LabelEncoder()
            if not istest:                     
                encoder.fit([row[iter] for row in (features+testdata)])
                np.save('./outputs/'+str(iter)+'_labelencoder.npy', encoder.classes_)
            else:
                encoder = featureencoders[iter]
            processedfeatures[:,iter] = np.array(encoder.transform([row[iter ] for row in features]))
        else:
            processedfeatures[:,iter] = np.array([row[iter] for row in features])


    #impute nans in numerical cols, auto excluding the categorical features as they have been replaced with classes already
    meanimputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    
    if not istest:        
        meanimputer.fit(processedfeatures)
        f = open("./outputs/meanimputer.pickle", "wb")
        pickle.dump(meanimputer,f )
        f.close()
    else:
        f = open('./outputs/meanimputer.pickle',"rb")
        meanimputer = pickle.load(f)
        f.close()
        
    processedfeatures_imp = meanimputer.transform(processedfeatures)
    return processedfeatures_imp


#############################################
# This is the main execution
# we read the train and test files 
# do feature preprocessing
# build model and do xval on train
# do out of sample val on test
#############################################
if __name__ == '__main__':
    filepath = './'
    trainfilename = 'TrainDataV6.csv'
    testfilename = 'TestDataV6.csv'
    
    # create the outputs folder
    os.makedirs('./outputs', exist_ok=True)

    labelcolumn = 3
    excludedcatcolumns = [labelcolumn] 
    excludednumcolumns = [] 
    categoricalfeaturestart = 0
    categoricalfeaturestartend = 3
    numtrees = 5

    #read from train data file
    fileread = readdata(filepath,trainfilename, labelcolumn,excludedcatcolumns, excludednumcolumns)
    traindata = fileread['data']
    trainlabel = fileread['label']

    #read from test data file    
    fileread = readdata(filepath,testfilename, labelcolumn, excludedcatcolumns, excludednumcolumns)
    testdata = fileread['data']
    testlabel = fileread['label']

    #process the train features
    processedtraindata_imp = processfeatures(traindata, categoricalfeaturestart, categoricalfeaturestartend, testdata)
    processedlabels = trainlabel

    #read in the cmd line argument if passed
    if len(sys.argv) > 1:
        numtrees = int(sys.argv[1])
    
    print("Number of trees {}.".format(numtrees))
    #perform random forest regression on train data
    from sklearn.ensemble import RandomForestRegressor
    rf_1 = RandomForestRegressor(random_state=0, n_estimators=numtrees)

    #output model
    r4 = rf_1.fit(processedtraindata_imp,processedlabels)
    pickle.dump(r4, open("./outputs/randomforest.pickle", "wb"))

    #output model performance
    score4 = cross_val_score(r4, processedtraindata_imp, processedlabels).mean()
    print("rf R-Sq on xval TRAIN = %.2f" % score4)

    from azureml.sdk import data_collector
    run_logger = data_collector.current_run() 

    # log the number of trees
    run_logger.log("Number of Trees", numtrees)

    # log rsquare
    run_logger.log("R-Square on xval train set", score4)

    #process the test features
    processedtestdata_imp = processfeatures(testdata, categoricalfeaturestart, categoricalfeaturestartend)
    processedtestlabels = testlabel

    #use the above model to predict on test data
    testpredictions = r4.predict(processedtestdata_imp)

    #calculate rmse
    testrmse = np.sqrt(np.mean(np.square(np.int64(testlabel) == np.int64(np.ceil(testpredictions)))))
    print('test RMSE = %.2f' % testrmse )

    #############################################
    # This code is used to do the visualization
    # it shows up under details for the particular run    
    #############################################
    import pandas
    from azureml.sdk import data_collector
    run_logger = data_collector.current_run() 
    metrics = []        
    for i in range(len(testlabel)):
        if i % 100 == 0:
            metrics.append({'Actual': testlabel[i]}) 

    run_logger.log(pandas.DataFrame(metrics))

    metrics2 = []
    for i in range(len(testpredictions)):
        if i % 100 == 0:
           metrics2.append({'Predicted': testpredictions[i]})  

    run_logger.log(pandas.DataFrame(metrics2))
