########################################################################################################################
# This program tests the init and run web service life cycle methods for bike forecasting 
# Inputs: expects the pickled/numpy files for regression model and tranforms and test data file (for simulating input)
#         for inputs to web service the columns should be as follows (in order, without the names/dictionary)
#         Hour	Weekday	start station id	RideInitiationCount	N_DryBulbTemp	N_RelativeHumidity	N_WindSpeed
# Outputs: the model prediction
########################################################################################################################


#############################################
# parser for reading off the test data file
# this is the same method as used in regression.py
#############################################

def readdata(filepath, filename, labelcolumn, excludedcatcolumns, excludednumcolumns):
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import preprocessing
    from sklearn.preprocessing import Imputer
    from sklearn.cross_validation import cross_val_score    
    import csv
    import os.path
    import pickle

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
# init is the lie cycle method for the web
# service that would be called once at the 
# initiialization of the web service
#############################################

def init():
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import preprocessing
    from sklearn.preprocessing import Imputer
    from sklearn.cross_validation import cross_val_score    
    import csv
    import os.path
    import pickle
    
    global categoricalfeaturestart
    global categoricalfeaturestartend
    global numcatfeatures

    categoricalfeaturestart = 0
    categoricalfeaturestartend = 3
    numcatfeatures = categoricalfeaturestartend
    
    # read in the model/transforms file and make them available globally for run method
    global model    
    model = pickle.load(open('./randomforest.pickle',"rb"))

    global meanimputer
    meanimputer = pickle.load(open('./meanimputer.pickle',"rb"))

    global featureencoders
    featureencoders = np.empty((numcatfeatures,), dtype=np.object)

    for iter in range(0, numcatfeatures):
        featureencoders[iter] = preprocessing.LabelEncoder()
        featureencoders[iter].classes_ = np.load('./'+str(iter)+'_labelencoder.npy')        

#############################################
# The run method is the life cycle method for
# web service that would be called each time
# a web request is received
#############################################
        
def run(inputString):
    import json
    import numpy as np
    try:
        input_list=json.loads(inputString)
    except ValueError:
        return 'Bad input: expecting a json encoded list of lists.'
    
    features = np.array(input_list)
    numfeatures = 6

    if (features.shape != (1, numfeatures)):
        return 'Bad input: Expecting a json encoded list of lists of shape (1,'+str(numfeatures)+').'
    
    #categorical to numerical transformatio
    processedfeatures = np.empty([len(features),numfeatures])

    for iter in range(0, numfeatures):
        if(iter >= categoricalfeaturestart and iter < categoricalfeaturestartend):                    
            processedfeatures[:,iter] = np.array(featureencoders[iter].transform([row[iter ] for row in features]))
        else:
            processedfeatures[:,iter] = np.array([row[iter] for row in features])


     #impute nans in numerical cols, auto excluding the categorical features as they have been replaced with classes already    
    processedfeatures_imp = meanimputer.transform(processedfeatures)
    
    #return predicted output    
    return str(np.int64(np.ceil(model.predict(processedfeatures_imp)[0])))


#############################################
# This is the main execution, it simulates the
# web service by calling init, and run
#############################################

import json
#perform initialization
init()

#read in the test data file
filepath = '.'
testfilename = 'TestDataV6.csv'
labelcolumn = 3
excludedcatcolumns = [labelcolumn] 
excludednumcolumns = []

fileread = readdata(filepath,testfilename, labelcolumn,excludedcatcolumns, excludednumcolumns)
testdata = fileread['data']

numfeatures = len(testdata[0])

#call run to do prediction, passing in last row from test data read above
print(run(json.dumps(testdata[-1:])))
