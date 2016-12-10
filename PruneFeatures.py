#!/usr/bin/env python
import sys
import pandas as pd
from TimeSeries import TimeSeries
from Image import Image
import numpy as np
import cPickle as pickle
import csv
from Data import Data
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import samples_generator
import logging, logging.config
from sklearn.cross_validation import train_test_split
from subprocess import call


class MultiData(Data):
    def __init__(self, name='', logConf='logging.conf'):
        super(MultiData, self).__init__(name, logConf)
    
    def readRawDataFromCSV(self, fname, hasY=True, sep=",", datatypesep="__", fnameSec = None, distanceFn = None, radius = 1000):
        """Reads data from a CSV file"""
        #Parse Primary dataset 
        print "reading file"
        xdata = []
        ydata = []
        with open(fname, 'rU') as f:
            
            r = csv.reader(f, delimiter=sep)
            header, types = super(MultiData, self).parseCSVHeader(r.next(), datatypesep)
            #print header
            for row in r:
                #print row
                if hasY:
                    #xdata.append(super(MultiData, self).parseRawRow(row[:-1], header, types))
                    xdata.append(super(MultiData, self).parseRawRow(row, header, types))
                    ydata.append(row[-1])
                else:
                    xdata.append(super(MultiData, self).parseRawRow(row, header, types))
               
        # Parse Secondary dataset
        
        if fnameSec is not None:
            print fnameSec
            with open(fnameSec, 'rU') as f:
                #print fnameSec
                r = csv.reader(f, delimiter=sep)
                #print header
                for _id, row in enumerate(r):
                    
                    #extract lat long information
                    primaryLat = float(row[1])
                    primaryLong = float(row[2])
                    for i, v in enumerate(row[3:]):
                        featureType = "LT_"+str(v.strip().split(":")[0])
                        secondLat = float(v.strip().split(":")[1])
                        secondLong = float(v.strip().split(":")[2])
                        #print _id, featureType, distanceFn(primaryLat, primaryLong, secondLat, secondLong)*1000
                        if (distanceFn(primaryLat, primaryLong, secondLat, secondLong)*1000 <= radius):
                            if featureType in xdata[_id]:
                                xdata[_id][featureType] += 1
                            else:
                                xdata[_id][featureType] = 1
        # extract features with enough non-missing data
        df = pd.DataFrame(xdata).dropna(axis=1, thresh = 5000).fillna(0)
        cols = list(df)
        #put predictor variable in the end for autoML
        cols.insert(len(cols)-1, cols.pop(cols.index('rodwycls')))
        df = df.ix[:,cols]
        #Write file to disk
        df.to_csv('ca_data', sep=',',index=False)
        ## Call R script to filter markov blanket and read features
        markovBlanket = calculateMrkovBlanket('ca_data')
        ##
        featuresDrop = [cols.index(feat) for feat in cols 
                        if feat not in markovBlanket 
                            or not feat.startswith("LT")]
        df = df.drop(df.columns[featuresDrop], axis=1)
        columns1 = map(lambda x : x+"__float", df.columns[:4])
        columns2 = map(lambda x : x+"__cat", df.columns[4:])
        
        df.columns = columns1 + columns2
        df.to_csv('ca_data_pruned', sep=',',index=False)
        
        sys.exit(0)
        
    
    def calculateMrkovBlanket(self, fname, node = "rdsurf"):
        features = {}
        call(["Rscript", "markovBlanket.r", fname])
        with open("edges.csv", r) as f:
            line = f.readLine().strip()
            if node in line:
                toAdd = []
                toAdd.append(line.split("\"")[1])
                toAdd.append(line.split("\"")[3])
                for feat in toAdd:
                    features[feat]
        return features
        
    
    def extendRawData(self, fname, data = None, hasY=False, sep=",", feature_sep = ":", header = None, types = None, 
                      distanceFn = None, distanceParams = None, radius = 1000):
        """Merge/Extend the original data with multiple datasets where extra datasets shares a 
        foreign keys. Extra data selection happens based upon the distance function that select
        elements in extra data corresponding to each row in the original dataset."""
        if data == None:
            print "can not extend data without original data"
            sys.exit(-1)
        features_dict = {}
        with open(fname, 'rU') as f:
            xdata = []
            ydata = []
            r = csv.reader(f, delimiter=sep)
            #header, types = self.parseCSVHeader(r.next(), datatypesep)
            #print header
            for row in r:
                #print row
                for entries in row.strip().split(sep):
                    features = entries.split(feature_sep)
                    if features[0] not in features_dict:
                        features_dict[features[0]] = 1
                    
                if hasY:
                    xdata.append(self.parseRawRow(row[:-1], header, types))
                    ydata.append(row[-1])
                else:
                    xdata.append(self.parseRawRow(row, header, types))
            self.processRawData(xdata, ydata, header, types)
                       
    

#########
if __name__ == '__main__':
    #d = Data("test")
    #d.readRawDataFromCSV("test.csv")
    #print d
    #d.exportToCSV('regtest.csv')
    g = Data("classification")
    g.generateClassificationData(n_features=2, n_samples=10)
    print g
    print g.addNoise()
    #h = Data("regression")
    #h.generateRegressionData()
    #print h
    #d = Data('comr.se')
    #d.readRawDataFromHive(host='13.4.40.80', username='bigdatafoundry', password=None,
    #                      database='bdf_prd_comrse_prd', table='transaction', maxRows=1000,
    #                      useCols=frozenset([2,3,4,5,9,11,12,13]))
    #print d
