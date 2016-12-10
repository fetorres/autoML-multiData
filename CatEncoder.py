#!/usr/bin/env python
from collections import OrderedDict
import logging, logging.config

class CatEncoder(object):
    def __init__(self, logConf='logging.conf'):
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.encoder = OrderedDict()
        self.revencoder = OrderedDict()
        self.num_categories = 0

    def fit(self, vec):
        """Create a mapping from the elements of the given vector to the range (0, number of categories)."""
        for entry in vec:
            if entry not in self.encoder:
                self.encoder[entry] = self.num_categories
                self.revencoder[self.num_categories] = entry
                self.num_categories += 1

    def transform(self, vec):
        """Transform the given vector based on the encoding learnt previously. If there is a new category then raise
        an exception."""
        return [self.encoder.get(v,v) for v in vec]

    def fit_transform(self, vec):
        """Transform the given vector based on the encoding learnt previously. If there is a new category, then add it
        to the encoder."""
        result = []
        for v in vec:
            try:
                encoding = self.encoder[v]
                result.append(encoding)
            except:
                #self.logger.debug('No mapping exists for %s - mapping it to %d' % (v, self.num_categories))
                #print('No mapping exists for %s - mapping it to %d' % (v, self.num_categories))
                self.encoder[v] = self.num_categories
                self.revencoder[self.num_categories] = v
                result.append(self.num_categories)
                self.num_categories += 1
        return result

    def getCategories(self):
        return self.encoder.keys()

    def isCategory(self, c):
        return c in self.encoder.keys()

    def __str__(self):
        return "CatEncoder: %d categories" % self.num_categories



##################
if __name__ == '__main__':
    a = ['a', 'b', 'a', 'd']
    b = ['a', 'b', 'a', 'd', 'e']
    ce = CatEncoder()
    ce.fit(a)
    print ce.transform(b)
    print ce
    print ce.getCategories()
