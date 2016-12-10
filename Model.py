#!/usr/bin/env python
# turn off the annoying scikit learn deprecation warnings
import warnings
def noop(*args, **kwargs):
    pass
warnings.showwarning = noop 
warnings.warn = noop
from Data import Data
from Experts import Experts
import signal
import platform
from time import time
## search, grid, cv, feature selection
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import scoreatpercentile
from sklearn.grid_search import RandomizedSearchCV
from RandomizedSearchCluster import RandomizedSearchCluster
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
## classifiers
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
## regressors
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
## outliers
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
#import logging, logging.config
## clustering
from ClusterWrapper import DbscanC, KmeansC

class Model(object):
    def __init__(self, data, test_size=0.25, logConf='logging.conf', n_pca_components=2):
        def signal_handler(signum, frame):
            raise Exception("Timed out!")
        global isUnixLike
        isUnixLike = platform.system().startswith('Linux') or platform.system().startswith('Darwin')
        if isUnixLike:  signal.signal(signal.SIGALRM, signal_handler)
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.data = data
        self.test_size = test_size
        self.experts = None
        self.n_pca_components = n_pca_components

    def trainClassifier(self, max_iterations, max_time=60, verbosity=0, excludeModels=[] ):
        """Train a supervised classification model. The parameters max_iterations and max_time can be used to
        place an upper limit on the grid search iterations and the time spent on each model fit. Finally,
        the excludeModels parameter can be used to specify a list of models that should not be fit (for example,
        if the user already knows that a particular model is unsuitable for a particular data set)."""
        self.nvars = self.data.X.shape[1]
        if self.nvars < 4:
            nfeatures = self.nvars
        else:
            nfeatures = max(1, 3 * self.nvars / 4)
        pca = PCA(self.n_pca_components)
        cselection = SelectKBest(f_classif, k=nfeatures)
        cfeatures = FeatureUnion([('pca', pca), ('kbest', cselection)])
        cfeatures_params = [('features__pca__n_components', sp_randint(1,nfeatures+1)), ('features__kbest__k', sp_randint(1, nfeatures+1))]
        
        self.cinit = {
            'AdaBoost': [Pipeline([('features', cfeatures), ('AdaBoost', AdaBoostClassifier(n_estimators=100))]),
                            dict(cfeatures_params + [('AdaBoost__learning_rate', sp_uniform(0.75, 1.25))])],
            'DecisionTree': [Pipeline([('features', cfeatures), ('DecisionTree', DecisionTreeClassifier())]),
                            dict(cfeatures_params +
                                 [('DecisionTree__max_depth', [3, 4, 5, None]), ('DecisionTree__max_features', ['sqrt', 'log2', None]),
                                    ('DecisionTree__min_samples_split', sp_randint(2, 11)), ('DecisionTree__min_samples_leaf', sp_randint(2, 11)),
                                    ('DecisionTree__criterion', ['gini', 'entropy'])])],
            'ExtraTrees': [Pipeline([('ExtraTrees', ExtraTreesClassifier())]),
                            dict(
                                 [('ExtraTrees__max_depth', [3, None]), #'ExtraTrees__max_features': sp_randint(1, nfeatures),
                                    ('ExtraTrees__min_samples_split', sp_randint(2, 11)), ('ExtraTrees__min_samples_leaf', sp_randint(1, 11)),
                                    ('ExtraTrees__bootstrap', [True, False]), ('ExtraTrees__criterion', ['gini', 'entropy'])])],
            'GaussianNB': [Pipeline([('features', cfeatures), ('GaussianNB', GaussianNB())]), dict(cfeatures_params)],

            'GradientBoost': [Pipeline([('features', cfeatures), ('GradientBoost', GradientBoostingClassifier(n_estimators=100))]),
                            dict(cfeatures_params +
                                 [('GradientBoost__max_depth', [3, None]), #'GradientBoost__max_features': sp_randint(1, nfeatures),
                                    ('GradientBoost__min_samples_split', sp_randint(2, 11)), ('GradientBoost__min_samples_leaf', sp_randint(1, 11)),
                                    ('GradientBoost__learning_rate', sp_uniform(0.01, 0.1))])],
            
            'KNeighbors': [Pipeline([('features', cfeatures), ('KNeighbors', KNeighborsClassifier())]),
                            dict(cfeatures_params +
                                 [('KNeighbors__n_neighbors', sp_randint(3, 15)), ('KNeighbors__weights', ['uniform', 'distance']),
                                    ('KNeighbors__p', sp_randint(1, 3))])],
            'LDA': [Pipeline([('features', cfeatures), ('LDA', LDA())]),
                    dict(cfeatures_params + [ ('LDA__n_components', sp_randint(1, self.nvars-1)) ] )],
            'LogisticRegression': [Pipeline([('features', cfeatures), ('LogisticRegression', LogisticRegression())]),
                    dict(cfeatures_params +
                         [('LogisticRegression__penalty', ['l1', 'l2']), ('LogisticRegression__C', sp_uniform(0.05, 5.0)),
                            ('LogisticRegression__fit_intercept', [True, False])])],
            'QDA': [Pipeline([('features', cfeatures), ('QDA', QDA())]),
                    dict(cfeatures_params + [('QDA__reg_param', sp_uniform(0.0, 1.0))])],
            'RandomForest': [Pipeline([('RandomForest', RandomForestClassifier(n_estimators=200))]),
                            dict(
                                 [('RandomForest__max_depth', [3, None]), ('RandomForest__n_estimators', sp_randint(10, 200)),
                                    ('RandomForest__min_samples_split', sp_randint(2, 11)),
                                    ('RandomForest__min_samples_leaf', sp_randint(1, 11)),
                                    ('RandomForest__bootstrap', [True, False]), ('RandomForest__criterion', ['gini', 'entropy'])])],
            'SGD': [Pipeline([('features', cfeatures), ('SGD', SGDClassifier())]),
                    dict(cfeatures_params +
                         [('SGD__loss', ['hinge', 'modified_huber', 'log', 'perceptron']),
                            ('SGD__penalty', ['l1','l2','elasticnet']), ('SGD__alpha', sp_uniform(0.0001, 0.0005)),
                            ('SGD__l1_ratio', sp_uniform(0.05, 0.95))])]
            #'''  SV causes autoML to get hung up on some datasets.  Even on Linux with the timeout working, it hangs.
            # PErhaps feature normalization would help.  For now, comment out SVM.
            #'SV': [Pipeline([('features', cfeatures), ('SV', SVC(C=1))]),
            #        dict(cfeatures_params + [('SV__kernel', ['linear', 'rbf']), ('SV__gamma', [1e-4, 1e-3]), ('SV__C', sp_uniform(0.2, 50))])]
            #'''
        }
        
        self.experts = Experts()
        
        for model in excludeModels:
            self.cinit.pop(model, None)
        max_model_time = max_time / len(self.cinit) 
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.getTrainAndTestData(test_size=self.test_size)
        self.trainSupervisedModel(self.getExecutionTime(self.cinit, max_model_time/2), self.cinit, max_iterations, max_model_time, verbosity)

    def trainRegressor(self, max_iterations, max_time=5*60, verbosity=0, excludeModels=[]):
        """Train the a supervised regression model. The parameters max_iterations and max_time can be used to
        place an upper limit on the grid search iterations and the time spent on each model fit. Finally,
        the excludeModels parameter can be used to specify a list of models that should not be fit (for example,
        if the user already knows that a particular model is unsuitable for a particular data set)."""
        self.nvars = self.data.X.shape[1]
        if self.nvars < 4:
            nfeatures = self.nvars
        else:
            nfeatures = max(1, 3 * self.nvars / 4)
        rselection = SelectKBest(f_regression, k=1)
        pca = PCA(self.n_pca_components, whiten=False)
        rfeatures = FeatureUnion([('pca', pca), ('kbest', rselection)])
        rfeatures_params = [('features__pca__n_components', sp_randint(1,nfeatures+1)), ('features__pca__whiten', [True, False]),
            ('features__kbest__k', sp_randint(1, nfeatures+1))]
        #rfeatures = FeatureUnion([('kbest', rselection)])
        #rfeatures_params = [('features__kbest__k', sp_randint(1, nfeatures))]
        self.rinit = {
            'AdaBoost': [Pipeline([('features', rfeatures), ('AdaBoost', AdaBoostRegressor(n_estimators=100))]),
                            dict(rfeatures_params + [('AdaBoost__learning_rate', sp_uniform(0.75, 1.25))])],
            'BayesianRidge': [Pipeline([('features', rfeatures), ('BayesianRidge', BayesianRidge())]),
                             dict(rfeatures_params +
                                  [('BayesianRidge__alpha_1', sp_uniform(1e-6, 1e-4)), ('BayesianRidge__alpha_2', sp_uniform(1e-6, 1e-4)),
                                    ('BayesianRidge__lambda_1', sp_uniform(1e-6, 1e-4)), ('BayesianRidge__lambda_2', sp_uniform(1e-6, 1e-4))])],
            'DecisionTree': [Pipeline([('features', rfeatures), ('DecisionTree', DecisionTreeRegressor())]),
                            dict(rfeatures_params +
                                 [('DecisionTree__max_depth', [3, 4, 5, None]), ('DecisionTree__max_features', ['sqrt', 'log2', None]),
                                    ('DecisionTree__min_samples_split', sp_randint(2, 11)), ('DecisionTree__min_samples_leaf', sp_randint(2, 11))])],
            'ElasticNet': [Pipeline([('features', rfeatures), ('ElasticNet', ElasticNet())]),
                            dict(rfeatures_params +
                                 [('ElasticNet__alpha', sp_uniform(0.5, 2.)), ('ElasticNet__l1_ratio', sp_uniform(0.,1.))])],
            'ExtraTrees': [Pipeline([('features', rfeatures), ('ExtraTrees', ExtraTreesRegressor())]),
                            dict(rfeatures_params +
                                 [('ExtraTrees__max_depth', [3, None]), #'ExtraTrees__max_features': sp_randint(1, nfeatures),
                                    ('ExtraTrees__min_samples_split', sp_randint(2, 11)), ('ExtraTrees__min_samples_leaf', sp_randint(1, 11)),
                                    ('ExtraTrees__bootstrap', [True, False])])],
            'GradientBoost': [Pipeline([('features', rfeatures), ('GradientBoost', GradientBoostingRegressor(n_estimators=100))]),
                            dict(rfeatures_params +
                                 [('GradientBoost__max_depth', [3, None]), #'GradientBoost__max_features': sp_randint(1, nfeatures),
                                    ('GradientBoost__min_samples_split', sp_randint(2, 11)),
                                    ('GradientBoost__min_samples_leaf', sp_randint(1, 11)),
                                    ('GradientBoost__learning_rate', sp_uniform(0.01, 0.1)),
                                    ('GradientBoost__loss', ['ls', 'lad', 'huber', 'quantile'])])],
            'KNeighbors': [Pipeline([('features', rfeatures), ('KNeighbors', KNeighborsRegressor())]),
                            dict(rfeatures_params +
                                 [('KNeighbors__n_neighbors', sp_randint(3, 15)), ('KNeighbors__weights', ['uniform', 'distance']),
                                    ('KNeighbors__p', sp_randint(0, 2))])],
            'Lasso': [Pipeline([('features', rfeatures), ('Lasso', Lasso())]),
                            dict(rfeatures_params + [('Lasso__alpha', sp_uniform(0.5,2.0))])],
            'RandomForest': [Pipeline([('features', rfeatures), ('RandomForest', RandomForestRegressor(n_estimators=200))]),
                            dict(rfeatures_params +
                                 [('RandomForest__max_depth', [3, None]), ('RandomForest__n_estimators', sp_randint(10, 200)),
                                    ('RandomForest__min_samples_split', sp_randint(2, 11)),
                                    ('RandomForest__min_samples_leaf', sp_randint(1, 11)),
                                    ('RandomForest__bootstrap', [True, False])])],
            'Ridge': [Pipeline([('features', rfeatures), ('Ridge', Ridge())]),
                            dict(rfeatures_params + [('Ridge__alpha', sp_uniform(0.5, 2.))])],
            'SGD': [Pipeline([('features', rfeatures), ('SGD', SGDRegressor())]),
                    dict(rfeatures_params +
                         [('SGD__loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                            ('SGD__penalty', ['l1','l2','elasticnet']), ('SGD__alpha', sp_uniform(0.0001, 0.0005)),
                            ('SGD__l1_ratio', sp_uniform(0.05, 0.95))])]
            #'''  SV causes autoML to get hung up on some datasets.  Even on Linux with the timeout working, it hangs.
            #'SV': [Pipeline([('features', rfeatures), ('SV', SVR(C=1))]),
            #        dict(rfeatures_params +
            #        [('SV__kernel', ['linear', 'rbf', 'poly', 'sigmoid']), ('SV__gamma', [1e-4, 1e-3]), ('SV__C', sp_randint(1, 1000)),
            #       ('SV__epsilon', sp_uniform(0.05, 0.2))])]
            #''' 
        }
        self.experts = Experts()
        
        for model in excludeModels:
            self.rinit.pop(model, None)
        max_model_time = max_time / len(self.rinit) 
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.getTrainAndTestData(test_size=self.test_size)
        self.trainSupervisedModel(self.getExecutionTime(self.rinit, max_model_time/10), self.rinit, max_iterations, max_model_time, verbosity)

    def trainSupervisedModel(self, model2exectime, init, max_iterations, max_model_time=60, verbosity=0):
        """This is the core method used for training supervised models. The parameters max_iterations and max_model_time can be used to
        place an upper limit on the grid search iterations and the time spent on each model fit. The parameter model2exectime
        is a map from the model name to the estimated execution time on the given data set. This estimate is used to compute the
        actual number of parameter grid searches that are actually performed (still limited on the top by max_iterations). """
        
        for model, duration in model2exectime.items():

            if isUnixLike:  signal.alarm( int( max_model_time ) + 1 )  # Start the timer - raise an exception if fitting this model does not complete in max_time seconds
            
            try: 
                v = init[model]
                n_iterations = min(max_iterations, int(max_model_time / duration))
                print("Fitting %s (n_iterations=%d, max_model_time=%ds)" % (model, n_iterations, max_model_time))
                t0 = time()
#                clf = RandomizedSearchCV(v[0], v[1], n_iter=n_iterations, verbose=verbosity, n_jobs=cpu_count())
                clf = RandomizedSearchCV(v[0], v[1], n_iter=n_iterations, verbose=verbosity) 
                clf.fit(self.X_train, self.y_train) 
                duration = time() - t0
                #self.logger.info("   Number of iterations: %d, Elapsed time: %0.2fs" % (n_iterations, duration))
                print("   Number of iterations: %d, Elapsed time: %0.2fs" % (n_iterations, duration))
                accuracy = clf.score(self.X_test, self.y_test)
                self.experts.insertExpert((accuracy, model, clf.best_estimator_, clf.best_params_))
            except Exception as msg :
                print("Skipping %s due to error: %s" % (model, msg))
                
        if isUnixLike:  signal.alarm(0)

    def getCVResults(self, clf):
        """Returns the out of the cross-validation."""
        #self.logger.debug("Best parameters set found on development set:")
        #self.logger.debug(clf.best_estimator_)
        y_pred = clf.predict(self.X_test)
        #self.logger.info(classification_report(self.y_test, y_pred))
        #self.logger.info(clf.score(self.X_test, self.y_test))
        print(classification_report(self.y_test, y_pred))
        print(clf.score(self.X_test, self.y_test))
        return accuracy_score(self.y_test, y_pred)

    def getExecutionTime(self, init, max_model_time=60, sample_size=3, verbosity=0):
        """This method is used to estimate the running time of a model fit prior to doing the full cross-validation run.
        The number of runs done for this estimation is specified by sample_size and the upper limit for doing those model
        fits is max_model_time."""
        execTime = {}
        for k, v in init.items():
            
            if isUnixLike: signal.alarm( int(max_model_time) + 1 ) # Interrupt the model fitting if it doesn't complete in max_time seconds
            
            try:
                t0 = time()
                print( "%s" % k )
                clf = RandomizedSearchCV(v[0], v[1], n_iter=sample_size, verbose=verbosity)
                clf.fit(self.X_train, self.y_train)
                duration = (time() - t0) + 1e-6 # avoid zero durations
                #self.logger.info("Time to fit %s instances of %s: %0.2fs" % (sample_size, k, duration))
                print("Time to fit %s instances of %s: %0.2fs" % (sample_size, k, duration))
                execTime[k] = duration / sample_size
            except Exception as msg :
                #self.logger.info("Skipping %s due to error: %s"%(k, msg))
                print("Skipping %s due to error: %s"%(k, msg))
                
        if isUnixLike:  signal.alarm(0)
        return execTime

    def trainOutlierDetector(self, max_percentile=10, max_model_time=60, max_iterations=10 ):
        """Build outlier detection models. The max_percentile parameter defines the maximum percentage of points that are
        defined as outliers while max_model_time specifies the maximum amount of time that can be spent on building each model.
        max_iterations is not used, but is present to be consistent with other methods
        """
        pca = PCA(self.n_pca_components)
        self.oinit = {
            'SV': Pipeline([('features', pca), ('SV', OneClassSVM())]),
            'EllipticEnvelope': Pipeline([('features', pca), ('EllipticEnvelope', EllipticEnvelope())])
        }
        self.experts = Experts()
        for model, clf in self.oinit.iteritems():
            if isUnixLike:  signal.alarm( int( max_model_time ) + 1 ) # Start the timer - raise an exception if fitting this model does not complete in max_time seconds
            try:
                #self.logger.info("Fitting %s (max_time=%ds)" % (model, max_time))
                print("Fitting %s (max_model_time=%ds)" % (model, max_model_time))
                t0 = time()
                clf.fit(self.data.X)
                duration = time() - t0
                #self.logger.info("   Elapsed time: %0.2fs" % duration)
                print("   Elapsed time: %0.2fs" % duration)
                distances = clf.decision_function(self.data.X).ravel()
                distance_threshold = scoreatpercentile(distances, max_percentile)
                self.experts.insertExpert((1.0, model, clf, distance_threshold))
            except Exception as msg :
                #self.logger.info("Skipping %s due to error: %s" % (model, msg))
                print("Skipping %s due to error: %s" % (model, msg))
        if isUnixLike: signal.alarm(0)     # Reset the timer before exiting the function

    def trainClusterer(self, max_iterations, max_time=5*60, verbosity=0, excludeModels=[]):
        """This is the core method used for learning clusters. The parameters max_iterations and max_time can be used to
        place an upper limit on the grid search iterations and the time spent on each model fit. The main work is done by
        by the RandomizedSearchCluster class which relies on IPython's parallel computing capabilties. As such, this method
        needs to specify an IPython client object that will be the interface to the parallel computing facilities."""
        self.clinit = {
            'Dbscan': [DbscanC(eps=0.5, min_samples=5),
                        {'eps': sp_uniform(1e-5, 4), 'min_samples': sp_randint(5, 30)}],
            'Kmeans': [KmeansC(n_clusters=2),
                    {'n_clusters': sp_randint(2, 20), 'init': ['k-means++', 'random'], 'n_init': [10]}]
        }
        #c = Client()
        #view = c.load_balanced_view()
        self.experts = Experts()

        max_model_time = max_time / ( len(self.cinit) - len( excludeModels ) ) + 1
        for model in excludeModels:
            self.clinit.pop(model, None)
        
        for model, v in self.clinit.iteritems():
            clf = None
            if isUnixLike: signal.alarm( int( max_model_time ) + 1 )  # Start the timer - raise an exception if fitting this model does not complete in max_time seconds
            try:
                #self.logger.info("Fitting %s (max_iterations=%d, max_time=%ds)" % (model, max_iterations, max_time))
                print("Fitting %s (max_iterations=%d, max_model_time=%ds)" % (model, max_iterations, max_model_time))
                t0 = time()
                clf = RandomizedSearchCluster(v[0], v[1], n_iter=max_iterations)
                clf.fit(self.data.X)  # serial search over the parameter space using IPython.parallel
                #clf.pfit(view, self.data.X)  # parallel search over the parameter space using IPython.parallel
                duration = time() - t0
                #self.logger.info("   Elapsed time: %0.2fs" % duration)
                print("   Elapsed time: %0.2fs" % duration)
                if clf.best_score > 0.0:  # it is possible that every iteration had an error - don't add to experts in that case
                    silhouette = clf.score(self.data.X, self.data.y)
                    self.experts.insertExpert((silhouette, model, clf.best_estimator, clf.best_estimator.get_params()))
            except Exception as msg :
                print("Stopping %s: %s" % (model, msg))
                #c.abort()
                if clf is not None and clf.best_score > 0.0:
                    duration = time() - t0
                    print("   Elapsed time: %0.2fs" % duration)
                    silhouette = clf.score(self.data.X, self.data.y)
                    self.experts.insertExpert((silhouette, model, clf.best_estimator, clf.best_estimator.get_params()))
                else:
                    print("No partial results available - skipping %s" % model)
        #c.close()
        if isUnixLike: signal.alarm(0)     # Reset the timer before exiting the function
        
    def predict(self, X, method='majority', **kwargs):
        """Use the trained models (i.e. the experts) to predict the outcome for the given data set. The combination of
        the outputs of the various models is specified with the 'method' parameter. By default, classification tasks
        use majority voting, regression tasks use averaging, and outlier tasks use conjunction."""
        return self.experts.predict(X, method, **kwargs)

    def score(self, X, y, method='majority', **kwargs):
        """For a supervised task, predict the result on the given dataset and return the performance metric (e.g.
        prediction accuracy for classification or R^2 for regression)."""
        ypred = self.experts.predict(X, method, **kwargs)
        return ypred
    
    def getExpert(self, name):
        try:
            return filter(lambda step: step[0]==name, filter(lambda expert: expert[1]==name, e.experts)[0][2].steps)[0][1]
        except:
            print("Expert %s not found" % name)
            return None
    

#########
if __name__ == '__main__':
    #blurt = Data('Blurt')
    #blurt.readRawDataFromCSV("blurt.csv", hasY=False)
    #gm = Model(blurt)
    #gm.trainClusterer(max_time=5*60, max_iterations=150)
    #print gm.experts
    #gm.experts.experts[0][2].plot(blurt.X, 'Blurt')
    #print gm.experts.predict(blurt.X)
    #doorA = Data('JRE Door A Current')
    #doorA.readRawDataFromCSV("jr_door_current.csv", hasY=False)
    #gm = Model(doorA)
    #gm.trainClusterer(max_time=15*60, max_iterations=150)
    #print gm.experts
    #gm.experts.experts[0][2].plot(doorA.X, title='JR Train Door A', pdffile='jr_train_door_clusters.pdf')
    #import matplotlib.pylab as pl
    #c = Data("MNIST Digits")
    #c.readRawDataFromCSV('../examples/data/digits.csv')
    #gm = Model(c)
    #gm.trainClassifier(max_time=60)
    #print gm.experts
    #print gm.experts.score(c.X, c.y)
    c = Data("classification")
    c.generateClassificationData()
    #c.saveToFile('test.classif.data.pickle')
    gm = Model(c)
    gm.trainClassifier()
    #gm.experts.saveToFile('test.classif.model.pickle')
    print ( gm.experts )
    print ( gm.experts.score(c.X, c.y) )
    #r = Data("regression")
    #r.generateRegressionData()
    #r.saveToFile('test.regress.data.pickle')
    #gm = Model(r)
    #gm.trainRegressor()
    #gm.experts.saveToFile('test.regress.model.pickle')
    #print gm.experts
    #print gm.experts.score(r.X, r.y, 'average')
    #ypred = gm.experts.predict(r.X, 'average')
    #r = Data("regression")
    #r.readRawDataFromCSV('../examples/data/output.txt')
    #print r
    #gm = Model(r, test_size=0.25)
    #gm.trainRegressor(max_iterations=500, max_time=5*60)
    #print gm.experts
    #print gm.experts.score(r.X, r.y, 'average')
    #ypred = gm.experts.predict(r.X, 'average')
    #pl.figure()
    #pl.plot(r.y, ypred, 'r.')
    #pl.show()
    #o = Data("Outlier Detection")
    #o.generateRegressionData(n_features=2)
    #gm = Model(o)
    #gm.trainOutlierDetector()
    #outliers = np.array(gm.predict(o.X, method='outlier'), dtype=bool)
    #print gm.experts
    #pl.figure()
    #pl.plot(o.X[outliers][:,0], o.X[outliers][:,1], 'ro')
    #pl.plot(o.X[:,0], o.X[:,1], 'b.')
    #pl.show()
