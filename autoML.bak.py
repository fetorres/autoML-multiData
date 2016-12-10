#!/usr/bin/env python

##  F Torres forked from S Kataria auotML-multiData on Aug 29, 2016
#####################################################################################
##  Imports
from Data import Data
from Model import Model
import subprocess
import platform
import re     #  for regular expressions
import argparse   # The argparse module makes it easy to write user-friendly command-line interfaces. 
#  sys provides access to some variables used or maintained by the interpreter and to functions that interact 
#  strongly with the interpreter. It is always available.  
import sys, os.path  
import numpy as np  #  NumPy is the fundamental package for scientific computing with Python. 
#import pickle
import pickle   #  For serializing and de-serializing a Python object structure, e.g. before dumping it to a file.  
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix

#####################################################################################
##  Default configuration of parameters

default_type2maxiterations = {'clustering': 10, 'classification': 100, 'regression': 100}
default_max_time = 1440   #  max time in sec for training all models.  Differs from original autoML.
n_pca_components_default = 2   # number of PCA components to use for feature selection
default_n_experts = 5   #number of experts to use in Ensemble scoring

## Default plots
def plot_confusion_matrix(cm, m, filename = 'testplot.png', title='Confusion matrix', cmap=plt.cm.Blues):
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm
    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(m.data.yClasses))
    plt.xticks(tick_marks, m.data.yClasses, rotation=45)
    plt.yticks(tick_marks, m.data.yClasses)
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    #plt.imsave('testplot.png', cm, cmap=cmap)
    plt.savefig(filename)
    
def plot_regression(y_test, ypred, m, filename = 'testplot.png', title='Regression Fit', cmap=plt.cm.Blues):
    
    plt.figure()
    sorted_ids = sorted(range(len(y_test)),key=lambda x:y_test[x])
    plt.plot(range(len(y_test)), sorted(y_test), c='k',  label='data')
    plt.plot(range(len(y_test)),[ypred[k] for k in sorted_ids], c='r', label='predictions')
    plt.title(title)
    plt.xlabel('data')
    plt.ylabel('taget')
    plt.legend(loc='lower right')
    
    #tick_marks = np.arange(len(m.data.yClasses))
    #plt.xticks(tick_marks, m.data.yClasses, rotation=45)
    #plt.yticks(tick_marks, m.data.yClasses)
    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.imsave('testplot.png', cm, cmap=cmap)
    plt.savefig(filename)

#####################################################################################
##  Agrument parser for command line interface

parser = argparse.ArgumentParser(
    description='A simple interface to autoML (lite, multiData)',
    epilog='It currently handles three types of models: classification, regression, and clustering. ' 
    + 'If the model type is classification or regression then the last column of the input data is assumed to be the dependent' 
    + ' variable.' 
)
parser.set_defaults(model_type='classification', input_file='data.csv', max_time=default_max_time, 
                    max_iterations=0, n_pca=n_pca_components_default, n_experts=default_n_experts,
                    sensitive_data_privatizing='manual', hierarchy_folder='hierarchy' )
parser.add_argument('-m', '--model_type', help = 'choose classification, regression or clustering' )
parser.add_argument('-i', '--input_file', help = 'primary input file to be analyzed' )
parser.add_argument('-t', '--max_time', help = 'maximum time in seconds for training all models. '
+ 'The default value is ' + str(default_max_time) +' seconds.' )
parser.add_argument('-n', '--max_iterations', type=int, 
                    help = 'max iterations for each individual model fit. The default is ' 
                    + str(default_type2maxiterations['clustering']) + ' for clustering, ' 
                    + str(default_type2maxiterations['classification']) + ' for classification and ' 
                    + str(default_type2maxiterations['regression']) + ' for regression.')
parser.add_argument('-pca', '--n_pca', type=int, help = 'number of PCA components' )
parser.add_argument('-e', '--n_experts', type=int, help = 'number of experts for Ensemble scoring' )                     

#  Check if data is sensitive.  For now, assume that only the first dataset can be sensitive.
parser.add_argument('-s', '--sensitive_data_privatizing', choices=['none', 'manual', 'auto'], help = 'choose none, manual, or auto for privatization of the data using ARX. For manual, an ARX window will launch.' )  
parser.add_argument('-f', '--hierarchy_folder', help = 'folder containing hierarchy files for sensitive data, if provided by the user' ) 

args = vars(parser.parse_args())
#if model_type is None or input_file is None:
#    parser.print_help()
#    parser.exit()

model_type =  args['model_type']
input_file = args['input_file']
files = [input_file]
max_time = args['max_time']
max_iterations = args['max_iterations']
n_pca_components = max( args['n_pca'], 2 )
n_experts = max( args['n_experts'], 2 )
privatize_data = args['sensitive_data_privatizing']
hierarchy_folder = args['hierarchy_folder']
if not os.path.exists(hierarchy_folder):
    os.makedirs(hierarchy_folder)
if max_iterations <= 0:
    max_iterations = default_type2maxiterations[model_type]

#####################################################################################
##  Run ARX for privatizing data if sensitive_data_privatizing != 'none'
##

files = []
if privatize_data == 'manual':
    ver = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
    ver64 = '64-bit' in str( ver ).lower()
    platform_system = platform.system().lower()
    
    if 'win' in platform_system:
        if ver64:
            subprocess.run( ['java', '-jar', 'arx_autoML_windows64.jar', hierarchy_folder, input_file] )
        else:
            subprocess.run( ['java', '-jar', 'arx_autoML_windows32.jar', hierarchy_folder, input_file] )
    elif 'linux' in  platform_system:
        if ver64:
            subprocess.run( ['java', '-jar', 'arx_autoML_linux64.jar', hierarchy_folder, input_file] )
        else:
            subprocess.run( ['java', '-jar', 'arx_autoML_linux32.jar', hierarchy_folder, input_file] )
    elif 'darwin' in platform_system:
        if ver64:
            subprocess.run( ['java', '-jar', 'arx_autoML_cocoa_macOSx_x86_64.jar', hierarchy_folder, input_file] )
        else:
            subprocess.run( ['java', '-jar', 'arx_autoML_cocoa_macOSx.jar', hierarchy_folder, input_file] )
    else:
        raise SystemExit('Unsupported OS: ' + platform_system + ' is not supported')
        
    with open(hierarchy_folder + '/output.txt', 'r') as f:
        files = [ line.strip() for line in f ]
    
elif privatize_data == 'auto':
    raise SystemExit('auto mode for privatizing data not yet implemented')


#####################################################################################
## Read the data file, create the Data() and Model() instances, and invoke the appropriate method.

if len(files) == 0:  
    files = [input_file]
    privatize_data = 'none'

for file in files:

    #  Print files saved by ARX, for debugging purposes only    
    #if privatize_data != 'none':
    #    fout = open(re.split("\.",file)[0] + '_config.txt', 'r')
    #    print ( fout.read() )
    #    fout.close
        
    if model_type == 'classification' or model_type == 'regression':
        hasY = True
    else:
        hasY = False   
    
    ##  Load previously computed models from pickle files, if they exist
    ##  Currently turned off by 'False' in the logic
    if (os.path.isfile('models.p')) and False:
        m = pickle.load( open( "models.p", "rb" ) )
        ypred = m.experts.predict(m.X_test, method='majority')
        plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, 'majority_voting', 'Confusion Matrix with Majority Voting')
        print ( len(m.experts.experts) )
        for i in range(5):
            ypred = m.experts.predict(m.X_test, method='majority', top = 1)
            plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, str(m.experts.experts[0][1])+"_"+str(i)+".png", 'Confusion Matrix with ' + str(m.experts.experts[0][1]) )
            del m.experts.experts[0]
        print ( "Ensemble Confusion Matrix (based on majority votes of top 5 models):\n%s\n%s" % (confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )
        sys.exit(0)
        
    if (os.path.isfile('regression_models.p')) and False:
        m = pickle.load( open( "regression_models.p", "rb" ) )
        ypred = m.experts.predict(m.X_test, method='average')
        plot_regression(m.y_test, ypred, m, 'majority_voting', 'Regression Fit w Averaging')
        
        print ( len(m.experts.experts) )
        for i in range(5):
            ypred = m.experts.predict(m.X_test, method='average', top = 1)
            
            plot_regression(m.y_test, ypred, m, str(m.experts.experts[0][1])+"_"+str(i)+".png", "Regression Fit w " + str(m.experts.experts[0][1]))
            del m.experts.experts[0]
        print ( "Ensemble Confusion Matrix (based on majority votes of top 5 models):\n%s\n%s" % (confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )
        sys.exit(0)
    
    ##  Load data	
    d = Data(file)
    d.readRawDataFromCSV(file, hasY)
    print ( d )
    model_type2method = {'clustering': 'trainClusterer', 'classification': 'trainClassifier', 'regression': 'trainRegressor'}
    m = Model(d, n_pca_components=n_pca_components )
    getattr(m, model_type2method[model_type])(max_time=max_time, max_iterations=max_iterations)
    print ( m.experts )
    m.experts.n_experts = n_experts
    #pickle.dump( m, open( "models.p", "wb" ))
    if model_type == 'classification':
        pickle.dump( m, open( "classify_models.p", "wb" ))
        ypred = m.experts.predict(m.X_test, method='majority')
        plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, 'majority_voting')
        
        #ypred = m.experts.predict(m.X_test, method='topcluster')
        #plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, m[0][2])
        print ( "Ensemble Confusion Matrix (based on majority votes of top 5 models):\n%s\n%s" % (confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )
        
    elif model_type == 'regression':
        pickle.dump( m, open( "regression_models.p", "wb" ))   # 'wb' means write only, binary mode (as opposed to text)
        ypred = m.experts.predict(m.X_test, method='average')
        print ( "Ensemble R^2 (based on weighted average of top 5 models): %0.3f\n" % r2_score(m.y_test, ypred) )
    	
    elif model_type == 'clustering':
        best_expert = m.experts.experts[0][2]
        secondary_results = '\n'.join(map(str, best_expert.labels_))
        #print secondary_results  # this can be stored in a file if needed - it shows the label associated with each data point
        best_expert.plot(d.X, title="Clustering of %s"%file, show=True)
