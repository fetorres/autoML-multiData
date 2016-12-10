#!/usr/bin/env python
from TimeSeries import TimeSeries
from Image import Image
import numpy as np
import pickle  # in Python 3.x, cpickle is considered a detail in implementing pickle, and not separately specified by users
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import samples_generator
#import logging, logging.config
from sklearn.cross_validation import train_test_split

class Data(object):
    def __init__(self, name='', logConf='logging.conf'):
        self.name = name
        self.X = None
        self.y = None
        self.Xnames = None
        self.yname = None
        self.xcorpora = {}
        self.vocabularies = {}
        self.ytype = None
        self.yEncoder = None  # use yEncoder.inverse_transform to recover the original category names
        self.yClasses = None
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')

    def generateClassificationData(self, n_samples=100, n_features=20, n_classes=2, n_informative=5, n_redundant=0, random_state=None):
        """Generate random data suitable for classification training tasks."""
        if n_informative > n_features:
            n_informative = n_features
        self.X, self.y = samples_generator.make_classification(n_samples, n_features,
                        n_informative, n_redundant, n_classes=n_classes, random_state=random_state)

    def generateRegressionData(self, n_samples=100, n_features=20, n_targets=1, n_informative=5, random_state=None):
        """Generate random data suitable for regression training tasks."""
        if n_informative > n_features:
            n_informative = n_features
        self.X, self.y = samples_generator.make_regression(n_samples, n_features,
                        n_informative, n_targets=n_targets, random_state=random_state)

    def generateClusteringData(self, n_samples=100, n_features=20, n_clusters=5):
        """Generate random data suitable for clustering tasks."""
        self.X, self.y = samples_generator.make_blobs(n_samples, n_features, n_clusters, random_state=None)

    def addNoise(self, level=0.1):
        """Add noise to non-categorical variables and return a matrix. The level of the noise is specified as a
        parameter. For each variable, the amount of noise is bouded by [(1-level)*value , (1+level)*value]"""
        Xnew = np.copy(self.X)   
        for i in range(Xnew.shape[1]):
            # Skip binary columns
            if not set(np.unique(Xnew[:,i])) == set([0,1]):
                noise = np.random.uniform(1-level, 1+level, Xnew.shape[0])
                Xnew[:,i] *= noise
        return Xnew
        
    def parseCSVHeader(self, header, datatypesep="__"):
        """Each column name is expected to be of the form <name><sep><type>.
        For example, height__float, age__int, race__cat represent three
        variables of types float, integer and category."""
        hd = [c.split(datatypesep) for c in header]
        # If the user hasn't specified the data type, then assume it to be a float
        for i in range(len(hd)):
            if len(hd[i]) == 1:
                hd[i].append('float')
        return [list(t) for t in zip(*hd)]

    def parseRawRow(self, row, header, types, ts_sep=';'):
        """X data is parsed and returned as a dictionary. Missing values are
        stored as np.nan. The returned data can later be processed with
        sklearn.feature_extraction.DictVectorizer() to handle categorical
        and missing variables."""
        rowDict = {}
        for i, v in enumerate(row):
            if len(v.strip()) > 0:
                if types[i] == "int":
                    rowDict[header[i]] =  round(float(v), 0)
                elif types[i] == "float":
                    rowDict[header[i]] =  float(v)
                elif types[i] == "ts":
                    # seems the following is more clear:  vals = [float(tsvs) for tsvs in v.split(ts_sep) if len(tsvs) > 0]
                    vals = [float(tsv) for tsv in [tsvs for tsvs in v.split(ts_sep) if len(tsvs) > 0]]
                    ts = TimeSeries(vals, 100, )
                    features = ts.getFeatures()
                    for j, f in enumerate(features):
                        rowDict["%s_%d" % (header[i], j)] = f
                elif types[i] == 'text':
                    if not i in self.xcorpora:
                        self.xcorpora[i] = []
                    self.xcorpora[i].append(v)
                elif types[i] == 'imgfile':
                    img = Image()
                    img.readFromFile(v)
                    features = img.getFeatures(10)
                    for j, f in enumerate(features):
                        rowDict["%s_%d" % (header[i], j)] = f
                else:
                    rowDict[header[i]] =  v
            else:
                rowDict[header[i]] = np.nan
        return rowDict

    def processRawData(self, xdata, ydata, header, types, stop_words='english'):
        """Instantiate self.X, self.y and related meta variables."""
        print ( "Converting file to features" )
        vec = DictVectorizer()
        self.X = vec.fit_transform(xdata).toarray()
        self.Xnames = vec.get_feature_names()
        for i, corpus in self.xcorpora.items():
            tfvec = TfidfVectorizer(stop_words=stop_words, vocabulary=self.vocabularies.get(i, None))
            self.X = np.hstack((self.X, tfvec.fit_transform(corpus).todense()))
            self.Xnames.extend(["%s_%s" % (header[i], f) for f in tfvec.get_feature_names()])
            self.vocabularies[i] = tfvec.get_feature_names()
        self.Xshape = self.X.shape
        if ydata:
            self.yname = header[-1]
            self.ytype = types[-1]
            if types[-1] == "cat":
                le = LabelEncoder()
                le.fit(ydata)
                self.y = le.transform(ydata)
                self.yEncoder = le
                self.yClasses = le.classes_
            else:
                self.y = np.array(map(float, ydata))

    def readRawDataFromCSV(self, fname, hasY=True, sep=",", datatypesep="__"):
        """Reads data from a CSV file"""
        with open(fname, 'rU') as f:
            xdata = []
            ydata = []
            r = csv.reader(f, delimiter=sep)
            header, types = self.parseCSVHeader(next(r), datatypesep)   # header=variable name for reach column, type=data type
            for row in r:      
                if hasY:
                    xdata.append(self.parseRawRow(row[:-1], header, types))   # returns a row dictionary 
                    ydata.append(row[-1])
                else:
                    xdata.append(self.parseRawRow(row, header, types))        # returns a row dictionary
            self.processRawData(xdata, ydata, header, types)
        self.types = types
        self.xdata = xdata
        self.header = header
                       
    def addRows(self, other):
        """Add rows using data from a different instance of Data()."""
        self.X = np.vstack((self.X, other.X))
        self.y = np.vstack((self.y, other.y))

    def addCols(self, other):
        """Add columns using data from a different instance of Data()."""
        self.X = np.hstack((self.X, other.X))
        for name in other.Xnames:
            self.Xnames.append( name )
        for vocab in other.vocabularies:
            self.vocabularies.update( vocab )

    def dropCols(self, indices):
        """Drop columns specified by the given indices."""
        self.X = np.delete(self.X, indices, axis=1)
        for i in sorted(indices, reverse=True):
            del self.Xnames[i]

    def saveToFile(self, fname):
        """Save the dataset as a Python pickle file."""
        pickle.dump({'name': self.name, 'X': self.X, 'y': self.y, 'Xnames': self.Xnames, 'yname': self.yname,
                     'ytype': self.ytype, 'yEncoder': self.yEncoder, 'yClasses': self.yClasses},
                    open(fname, "wb"))

    def exportToCSV(self, fname):
        """Save the dataset as a comma separated text file."""
        with open(fname, 'wb') as f:
            cf = csv.writer(f)
            if self.y is None:
                cf.writerow(self.Xnames)
            else:
                cf.writerow(self.Xnames + [self.yname])
            for i, row in enumerate(self.X):
                if self.y is None:
                    cf.writerow(row)
                else:
                    cf.writerow(np.hstack((row, self.y[i])))

    def loadFromFile(self, fname):
        d = pickle.load(open(fname, "rb"))
        self.name = d['name']
        self.X = d['X']
        self.y = d['y']
        self.Xnames = d['Xnames']
        self.yname = d['yname']
        self.ytype = d['ytype']
        self.yEncoder = d['yEncoder']
        self.yClasses = d['yClasses']

    def getTrainAndTestData(self, test_size=0.45, random_state = 1):
        if self.y is not None:
            return train_test_split(self.X, self.y, test_size=test_size, random_state = random_state)
        else:
            return train_test_split(self.X, test_size=test_size, random_state = random_state)

    def __str__(self):
        if self.y is None and self.X is not None:
            return(
        """Dataset '%s': %s
        Column names: %s
        Row 1: %s
        Row -1: %s""" % (self.name, self.X.shape, self.Xnames, self.X[0, :], self.X[-1, :]))
        elif self.X is not None:
            return(
        """Dataset '%s': %s
        Column names: %s
        Target name: %s
        Target type: %s
        Target classes: %s
        Target encoding: %s
        Row 1: %s -> %s
        Row -1: %s -> %s""" % (self.name, self.X.shape, self.Xnames, self.yname, self.ytype, self.yClasses,
                               (self.yEncoder.transform(self.yClasses) if self.yEncoder else ''),
                               self.X[0, :], self.y[0], self.X[-1, :], self.y[-1]))
        else:
            return("Empty Dataset '%s'" % self.name)

#########
if __name__ == '__main__':
    #d = Data("test")
    #d.readRawDataFromCSV("test.csv")
    #print d
    #d.exportToCSV('regtest.csv')
    g = Data("classification")
    g.generateClassificationData(n_features=2, n_samples=10)
    print ( g )
    print ( g.addNoise() )
    #h = Data("regression")
    #h.generateRegressionData()
    #print h
    #d = Data('comr.se')
    #d.readRawDataFromHive(host='13.4.40.80', username='bigdatafoundry', password=None,
    #                      database='bdf_prd_comrse_prd', table='transaction', maxRows=1000,
    #                      useCols=frozenset([2,3,4,5,9,11,12,13]))
    #print d
