#!/usr/bin/env python
import bisect
import pickle
import numpy as np
## search, grid, cv, feature selection
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix
#import logging, logging.config

class Experts(object):
    def __init__(self, logConf='logging.conf'):
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.experts = []

    def loadFromFile(self, filename):
        """Read pickled models from the specified file."""
        self.experts = pickle.load(open(filename, 'rb'))

    def saveToFile(self, filename):
        """Save the models as a Python pickle file."""
        pickle.dump(self.experts, open(filename, 'wb'))

    def insertExpert(self, e):
        """Insert an expert of the form (accuracy, modelType, model, parameters)."""
        bisect.insort(self.experts, (1. - e[0], e[1], e[2], e[3]))

    def predict(self, X, method='majority', **kwargs):
        """Return the prediction of the top models."""
        if self.experts:
            top = kwargs.get('top', 5)
            if method == 'topcluster':
                self.experts[0][2].score(X)
            else:
                predictions = np.transpose(np.array([e[2].predict(X) for e in self.experts[:top]]))
            return [self.voting(p, method, **kwargs) for p in predictions]
        else:
            raise Exception("No experts exist")

    def score(self, X, y, method='majority', **kwargs):
        """Return the accuracy of the experts on X against the actual value y"""
        if method == 'majority':
            ypred = self.predict(X, method, **kwargs)
            #self.logger.info(confusion_matrix(y, ypred))
            #self.logger.info(classification_report(y, ypred))
            return accuracy_score(y, ypred)
        elif method == 'average':
            return r2_score(y, self.predict(X, method, **kwargs))
        elif method == 'topcluster':
            return self.experts[0][2].score(X, y)
        else:
            raise Exception("Unknown voting method: %s" % method)

    def voting(self, predictions, method='majority', **kwargs):
        """Compute the prediction by combining the output of the experts (i.e. trained models). The voting method is specified
        by an input parameter - currently, the recognized methods are 'majority', 'average', and 'outlier'. For each of the methods,
        other parameters may be specified using the kwargs dictionary. For example, for the majority voting, 'threshold' defines the
        number of votes needed to override the default output (which is specified by the 'default' parameter). If 'threshold' is not
        defined then a simple majority is required. If 'default' is not specified then it is set to be the output of the top model."""
        if method == 'majority':
            threshold = kwargs.get('threshold', len(self.experts) / 2)  # if no threshold specified then use 50%
            default = kwargs.get('default', predictions[0])   # if no default value specified then use the view of the top expert
            counts = np.bincount(predictions)
            i = np.nonzero(counts)[0]
            hist = list( zip(counts[i], i) )
            if hist[0][0] >= threshold:
                return hist[0][1]       # return the majority view if one exists (i.e. top count > threshold)
            else:
                return default          # return the default answer if no majority view exists
        elif method == 'average':
            return np.mean(predictions)
        elif method == 'outlier':
            threshold = kwargs.get('threshold', len(self.experts))  # if no threshold specified then declare outlier if all agree
            return (np.array(len(self.experts)*[-1]) == predictions).sum() >= threshold
        else:
            raise Exception("Unknown voting method: %s" % method)

    def getNumberOfExperts(self):
        return len(self.experts)

    def __str__(self):
        return(
        """
        Number of models: %d
        Models: %s
        """ % ( len(self.experts), ["%s: %f" % (e[1], 1. - e[0]) for e in self.experts] ) )
    
    def getDetails(self):
        result = ""
        for e in self.experts:
            result += """%s\n%s:
            Accuracy: %f
            Parameters for best fit: %s
            Pipeline: %s\n""" % ('-'*40, e[1], 1.-e[0], e[3], e[2].steps)
        return result       
    


###########
if __name__ == '__main__':
    #from Data import Data
    e = Experts()
    e.loadFromFile('../examples/data/test.experts.pickle')
    print ( e.getDetails() )
    #d = Data()
    #d.loadFromFile('test.data.pickle')
    #print "Accuracy as per the ensemble:", e.score(d.X, d.y)
    #print "Accuracy as per best expert:", accuracy_score(e.experts[0][2].predict(d.X), d.y)
    
