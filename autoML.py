#!/usr/bin/env python

##  F Torres forked from S Kataria auotML-multiData on Aug 29, 2016
#####################################################################################
##  Imports
from Data import Data
from Model import Model
from DeIdentify import DeIdentify
from FuseData import FuseData
import argparse   # The argparse module makes it easy to write user-friendly command-line interfaces. 
#  sys provides access to some variables used or maintained by the interpreter and to functions that interact 
#  strongly with the interpreter. It is always available.  
import sys, os.path  
import numpy as np  #  NumPy is the fundamental package for scientific computing with Python. 
#import pickle
import pickle   #  For serializing and de-serializing a Python object structure, e.g. before dumping it to a file.  
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, r2_score, confusion_matrix

class p:   # namespace for command line parser
    pass

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
##  Default configuration of parameters
#
# default_max_time is max time in sec for training all models.  Differs from original autoML.
# n_pca_components_default is the number of PCA components to use for feature selection in outlier detection.
#       The value is over-ridden during cross validation for classification, clustering, and regression.
# 

default_type2maxiterations = {'clustering': 10, 'classification': 100, 'regression': 100}
default_max_time = 1440
n_pca_components_default = 4
default_n_experts = 5   #number of experts to use in Ensemble scoring

# Try putting all the setup code in a function

def autoML_setup():


    #####################################################################################

    parser = argparse.ArgumentParser(
        description='A simple interface to autoML (lite, multiData)',
        epilog='It currently handles four types of models: classification, regression, clustering and outlier detection. ' 
        + 'If the model type is classification or regression then the last column of the input data is assumed to be the dependent' 
        + ' variable.' 
        + ' Option to add a second dataset and a distance function: The distance function is used to assign elements of the' 
        + ' second dataset to each row in the first dataset.  A cutoff radius is used for the selection, with default initial value'
        + ' of 1. The -r option can be used to scale the distance function differently.'
    )

    helpText={}
    helpText[ 'model_type' ] = 'choose classification(default), regression, clustering or outlier_detection'
    helpText[ 'input_file' ] = 'primary input file to be analyzed (default=data.csv)'
    helpText[ 'secondary_file' ] = 'optional secondary input file, triggers multi-dataset analysis (default=None)'
    helpText[ 'secondary_weights' ] = 'weights for features in secondary file, default=False'
    helpText[ 'distanceFn' ] = 'choose L_1Norm(n), euclidean(n), L_infinityNorm(n), distanceOnEarth(n),' + \
                        ' L_1Norm_cat(n), or L_infinityNorm_cat(n), where n=1,2,3,... is the chosen dimension' + \
                        ' for calculating distances.  Default is L_1Norm(1)'
    helpText[ 'sparsity' ] = 'sparsity threshold for including records in secondary input file'
    helpText[ 'radius' ] = 'radius for cutoff of the distance function (default=1)'
    helpText[ 'max_time' ] = 'maximum time in seconds for training all models. '+ \
                                'The default value is ' + str(default_max_time) +' seconds.'
    helpText[ 'max_iterations' ] = 'max iterations for cross-validation of each individual model fit. The default is ' \
                        + str(default_type2maxiterations['clustering']) + ' for clustering, ' \
                        + str(default_type2maxiterations['classification']) + ' for classification and ' \
                        + str(default_type2maxiterations['regression']) + ' for regression.'
    helpText[ 'n_pca' ] = 'number of PCA components for outlier detection.  (default is ' + str(n_pca_components_default) + ')' 
    helpText[ 'n_experts'] = 'number of experts for Ensemble scoring.  (default is ' + str(default_n_experts) + ')' 
    helpText[ 'privatize_data' ] = 'choose none, manual, or auto for privatization of the data using ARX. For manual, an ARX window ' + \
                        'will launch. For privatization of primary and secondary datasets, choose manual_both or auto_both.' \
                        + '  Default is ''manual''.'
    helpText[ 'hierarchy_folder' ] = 'folder containing hierarchy files for sensitive data, if provided by the user.' \
                                                            + '  Default is ''hierarchy'
    helpText[ 'regression_test' ] = 'when set, execute regression test rather than ordinary operation'''

    parser.set_defaults(model_type='classification', input_file='data.csv', secondary_file = None, secondary_weights = False,
                        distanceFn='L_1Norm(1)', radius=1, max_time=default_max_time, max_iterations=0, 
                        n_pca=n_pca_components_default, n_experts=default_n_experts,
                        privatize_data='manual', hierarchy_folder='hierarchy', regression_test=False )

    parser.add_argument('-m', '--model_type', help = helpText[ 'model_type' ] )
    parser.add_argument('-i', '--input_file', help = helpText[ 'input_file' ] )
    parser.add_argument('-file2', '--secondary_file', help = helpText[ 'secondary_file' ] )
    parser.add_argument('-w', '--secondary_weights', type=bool, help = helpText[ 'secondary_weights' ] )
    parser.add_argument('-d', '--distanceFn', help = helpText[ 'distanceFn' ] )
    parser.add_argument('-sprs', '--sparsity', type=float, help = helpText[ 'sparsity' ] )
    parser.add_argument('-r', '--radius', type=float, help = helpText[ 'radius' ] )
    parser.add_argument('-t', '--max_time', type=float, help = helpText[ 'max_time' ] )
    parser.add_argument('-n', '--max_iterations', type=int, help = helpText[ 'max_iterations' ] )
    parser.add_argument('-pca', '--n_pca', type=int, help = helpText[ 'n_pca' ] )
    parser.add_argument('-e', '--n_experts', type=int, help = helpText[ 'n_experts'] )                     
    parser.add_argument('-s', '--privatize_data', choices=['none', 'manual', 'auto', 'manual_both', 'auto_both'], help = helpText[ 'privatize_data' ] )  
    parser.add_argument('-f', '--hierarchy_folder', help = helpText[ 'hierarchy_folder' ] )
    parser.add_argument('-x', '--regression_test', help = helpText[ 'regression_test' ] )

    return parser

# That's the end of autoML_setup.  At this point, the caller parses the arguments, then
# calls autoML_process.

# Here's where we do the real processing
def autoML_process(p):

    # if we're performing the regression test, initialize one of
    # the sources of random numbers.
    if p.regression_test:
        np.random.seed(0xDEADBEEF)

    #if model_type is None or input_file is None:
    #    parser.print_help()
    #    parser.exit()

    n_experts = max( p.n_experts, 2 )

    distanceFn = p.distanceFn.split( '(' )[0]
    distanceFn_dim = int( p.distanceFn.split( '(' )[1].split( ')' )[0] )

    if not os.path.exists(p.hierarchy_folder):
        os.makedirs(p.hierarchy_folder)
    if p.max_iterations <= 0:
        p.max_iterations = default_type2maxiterations[p.model_type]

    #####################################################################################
    ##  Create privatized data files if sensitive_data_privatizing != 'none'.
    ##  Otherwise, the DeIdentify class will use input file as is.
    ##

    privatizeBoth = ( p.privatize_data.split('_')[-1] == 'both' )
    privatize_data = p.privatize_data.split('_')[0]    # Remove the '_both' from privatize_data
    if ( privatizeBoth ):
        safe_secondary = DeIdentify( p.secondary_file, p.hierarchy_folder )
        safe_secondary.deIdentify( privatize_data )

    safe = DeIdentify( p.input_file, p.hierarchy_folder )
    safe.deIdentify( privatize_data )

    #####################################################################################
    ## Read the data file, create the Data() and Model() instances, and invoke the appropriate method.

    for file in safe.desensitizedFiles:

        if p.model_type == 'classification' or p.model_type == 'regression':
            hasY = True
        else:
            hasY = False   

        ##  Load previously computed models from pickle files, if they exist
        ##  Currently turned off by 'False' in the logic
        if (os.path.isfile('classify_models.p')) and False:
            m = pickle.load( open( "models.p", "rb" ) )
            ypred = m.experts.predict(m.X_test, method='majority', top=n_experts)
            plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, 'majority_voting', 'Confusion Matrix with Majority Voting')
            print ( len(m.experts.experts) )
            for i in range( n_experts ):
                ypred = m.experts.predict(m.X_test, method='majority', top = 1)
                plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, str(m.experts.experts[0][1])+"_"+str(i)+".png", 'Confusion Matrix with ' + str(m.experts.experts[0][1]) )
                del m.experts.experts[0]
            print ( "Ensemble Confusion Matrix (based on majority votes of top %d models): \n%s\n%s" % (n_experts, confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )
            sys.exit(0)

        if (os.path.isfile('regression_models.p')) and False:
            m = pickle.load( open( "regression_models.p", "rb" ) )
            ypred = m.experts.predict(m.X_test, method='average', top=n_experts)
            plot_regression(m.y_test, ypred, m, 'majority_voting', 'Regression Fit w Averaging')

            print ( len(m.experts.experts) )
            for i in range(n_experts):
                ypred = m.experts.predict(m.X_test, method='average', top = 1)

                plot_regression(m.y_test, ypred, m, str(m.experts.experts[0][1])+"_"+str(i)+".png", "Regression Fit w " + str(m.experts.experts[0][1]))
                del m.experts.experts[0]
            print ( "Ensemble Confusion Matrix (based on majority votes of top %d models):\n%s\n%s" % (n_experts, confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )
            sys.exit(0)

        ##  Load data	
        d = Data(file)
        d.readRawDataFromCSV(file, hasY)

        ##  Fuse secondary data if appropriate
        if p.secondary_file :
            addData = FuseData( d, distanceFn, p.secondary_weights )
            addData.radius = p.radius
            addData.fuse( p.secondary_file, distanceFn_dim )
            d = addData.data1

        print(d)

        ##  Train models
        model_type2method = {'clustering': 'trainClusterer', 'classification': 'trainClassifier', 'regression': 'trainRegressor',
                             'outlier_detection': 'trainOutlierDetector'}
        if p.regression_test :
            m = Model(d, n_pca_components=max( p.n_pca, 2 ), do_regression_test=True )
        else:
            m = Model(d, n_pca_components=max( p.n_pca, 2 ) )
        getattr(m, model_type2method[p.model_type])( max_time=p.max_time, max_iterations=p.max_iterations )

        # we've just finished fitting each classifier, regressor or clusterer to the data.
        # Return the model so that we can evaluate the results.
        return (m, n_experts)

def run_autoML() :
    """ Provide an entry point to run autoML """
    parser = autoML_setup()
    parser.parse_args(namespace=p)
    (m, n_experts) = autoML_process(p)

    # at this point, all the specified models have been fitted.  Print out
    # the list of classifiers, regressors or clusterers and
    # the figure of merit for each one
    print ( m.experts )

    ##  Test models if method is classification or regression, show results for clustering
    if p.model_type == 'classification':
        pickle.dump( m, open( "classify_models.p", "wb" ))
        ypred = m.experts.predict(m.X_test, method='majority', top=n_experts)
        plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, 'majority_voting')

        #ypred = m.experts.predict(m.X_test, method='topcluster', top=n_experts)
        #plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, m[0][2])
        print ( "Ensemble Confusion Matrix (based on majority votes of top %d models):\n%s\n%s" % (n_experts, confusion_matrix(m.y_test, ypred), classification_report(m.y_test, ypred)) )

    elif p.model_type == 'regression':
        pickle.dump( m, open( "regression_models.p", "wb" ))   # 'wb' means write only, binary mode (as opposed to text)
        ypred = m.experts.predict(m.X_test, method='average', top=n_experts)
        print ( "Ensemble R^2 (based on weighted average of top %d models): %0.3f\n" % (n_experts, r2_score(m.y_test, ypred) ) )

    elif p.model_type == 'clustering':
        best_expert = m.experts.experts[0][2]
        secondary_results = '\n'.join(map(str, best_expert.labels_))
        #print secondary_results  # this can be stored in a file if needed - it shows the label associated with each data point
        best_expert.plot(d.X, title="Clustering of %s"%file, show=True)

# Here's where the automatic testing code feeds in faux parser input and checks the output strings
def test_autoML(commandLineArgs=[]) :
    print("Starting test_autoML_outputs")
    # set things up
    parser = autoML_setup()

    parser.parse_args(args=commandLineArgs, namespace=p)
    (m, n_experts) = autoML_process(p)

    # compare the outputs
    if p.model_type == 'classification':
        pickle.dump( m, open( "classify_models.p", "wb" ))
        ypred = m.experts.predict(m.X_test, method='majority', top=n_experts)
        plot_confusion_matrix(confusion_matrix(m.y_test, ypred), m, 'majority_voting')

        # return the three strings that constitute our output.
        # These are the classifiers with their figures of merit,
        # a confusion matrix and a report on how well the classifiers did.
        return {'experts':str(m.experts),
                'confusionMatrix':str(confusion_matrix(m.y_test, ypred)),
                'classificationReport':str(classification_report(m.y_test, ypred))}
    elif p.model_type == 'regression':
        pickle.dump( m, open( "regression_models.p", "wb" ))   # 'wb' means write only, binary mode (as opposed to text)
        ypred = m.experts.predict(m.X_test, method='average', top=n_experts)
        return {'experts':str(m.experts),
                'regressionReport':(n_experts, r2_score(m.y_test, ypred))}
    elif p.model_type == 'clustering':
        print("Haven't implemented regression test for clustering")
    elif p.model_type == 'outlier_detection':
        print("Haven't implemented regression test for outlier_detection")


# here's where we say what should be executed when this module is the "main" module, as opposed to a module that's
# loaded by test_autoML.py.
if __name__ == '__main__':
    run_autoML()

