#!/usr/bin/env python
from sklearn.grid_search import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import logging, logging.config
#from parallel import est_fit
#from IPython.parallel import Client
from time import time
from ClusterWrapper import DbscanC, KmeansC

class RandomizedSearchCluster(object):
    def __init__(self, estimator, param_distributions, n_iter=70, random_state=None, logConf='logging.conf'):
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_estimator = estimator
        self.best_score = 0.0

    def fit(self, train):
        """Run fit on the estimator with randomly drawn parameters. Runs sequentially.
        
        Parameters
        ----------
        train: training data
        """
        self.best_score = 0.0
        sampled_params = ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
        for parameters in sampled_params:
            try:
                estimator = self.estimator.clone()
                estimator.set_params(**parameters)
                print('%s' % estimator)
                estimator.fit(train)
                score = estimator.score(train, None)
                #self.logger.debug('Average silhouette score for this estimator: %0.2f' % score)
                print('  Average silhouette score for this iteration: %0.2f' % score)
                unique_labels = set(estimator.labels_)
                num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                print('  Estimated clusters: %d' % num_clusters)
                if score > self.best_score:
                    self.best_score = score
                    self.best_estimator = estimator
            except ( Exception, e ):
                print("Skipping iteration due to error: %s" % e)
    
    #def pfit(self, view, train):
    #    """Run fit on the estimator with randomly drawn parameters. Runs in parallel using IPython.parallel methods.
    #    
    #    Parameters
    #    ----------
    #    view: IPython view on which to perform the computation
    #    train: training data
    #    """
    #    self.best_score = 0.0
    #    sampled_params = ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
    #    estimators = view.map_async(est_fit, [(train, self.estimator.clone().set_params(**p)) for p in sampled_params], ordered=False)
    #    for estimator in estimators:
    #        score = estimator.score(train, None)
    #        #self.logger.debug('Average silhouette score for this estimator: %0.2f' % score)
    #        print('  Average silhouette score for this iteration: %0.2f' % score)
    #        if score > self.best_score:
    #            self.best_score = score
    #            self.best_estimator = estimator

    def score(self, test, true_labels):
        """Return the score of the best estimator.

        Parameters
        ----------
        test: test data
        true_labels: (optional) true labels if available
        """
        return self.best_estimator.score(test, true_labels)

    def __str__(self):
        return "%s" % self.best_estimator


####################
if __name__ == '__main__':
    pass
