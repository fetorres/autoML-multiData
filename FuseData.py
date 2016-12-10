# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 22:36:32 2016

@author: torres
class for fusing data from a secondary file extracted
from an auxiliary dataset into the primary data.  The fused data consists of 
new columns, one for each new element.  The value of each column is the 
number of elements of that column type that are within a specified cutoff radius
of the primary data row.  For example, a column could be "Library" and the values 
could be counts from 1 to the upper limit, for a radius of 2 km.
There is a lot of flexibility to define what is meant by distance.

The secondary data csv file must be in the format expected by autoML,
which is described in 'UsingDistanceFunctionToJoinDisparateDatasets.pdf'

"""
from Data import Data
from DistanceFn import DistanceFn
import csv
import numpy as np
#import logging, logging.config

class FuseData(object):
    def __init__(self, Data1, distanceFn, weights, logConf='logging.conf'):
        self.data1 = Data1        
        self.distanceFn = DistanceFn(distanceFn)
        self.weights = weights
                
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')

    def fuse(self, file2, dim, sep=',' ):
        """Reads secondary data from a CSV file.
        """
        
        self.dim = dim
        
        with open(file2, 'rU') as csvfile:    # r=open for reading, U=deprecated universal newlines mode
            r = csv.reader( csvfile, delimiter=sep )
            a = next(r)   # skip header row

            self.fuseType = 'RowByRow'
            self.fuseRowByRow( r )
        
    def fuseRowByRow(self, r ):
        ''' The distance from a row of self.Data to each element in the same 
            row of a secondary file is computed.  For those that fall within 
            self.radius, the count in the primary data is increased.
            
            Preconditions:
            r is a file reader object which will iterate over lines in the 
                secondary file.
        '''
        
        xdata = []
        header = []
        
        for id, row in enumerate(r): 
            
            # load into vector1 the data from data1 for each column used in a distance calculation.
            # The distance vector is defined to be the first self.dim columns of data1.xdata
            vector1 = [ self.data1.xdata[id][ self.data1.header[ih] ]  for ih in range( self.dim ) ]           
            
            # need to change data type from string to int or float or NaN when appropriate             
            for ih in range( self.dim ):
                if vector1[ih] :
                    if self.data1.types[ih] == "int":
                        vector1[ih] =  round(float(vector1[ih]), 0)
                    elif self.data1.types[ih] == "float":
                        vector1[ih] =  float(vector1[ih])
                else:
                    vector1[ih] = np.nan  # makes robust to missing data in the secondary data file

            '''
            candidates in the secondary csv file are semicolon-delimited lists,
            with the first item in the list being an ID of some sort.
            If there is not at least one additional item, then the element is a blank candidate,
            and therefore is not used.  If the candidate semicolon-delimted list 
            has values beyond the ID, then there should be at least self.dim of them, 
            since that is the length of the distance vectors.
            See UsingDistanceFunctionToJoinDisparateDatasets.pdf
            '''
            rowDict={}    # initialize row dictionary
            for col in row[1:]:          # skips first column, since it is just identifying info for the row
                colParsed = col.strip().split(":")    # parse elements separated by colons
                
                if len(colParsed) > self.dim :                
                    featureName = "FusedFeature_"+str(colParsed[0])
                    vector2 = colParsed[ 1 : self.dim+1 ]
                    
                    for ix in range( self.dim ):   #  change variables from string if column type is int or float
                        if len(vector2[ix]) > 0:   
                            if self.data1.types[ix] == "int":
                                vector2[ix] =  round(float(vector2[ix]), 0)
                            elif self.data1.types[ix] == "float":
                                vector2[ix] =  float(vector2[ix])

                    # now check if element is within the specified radius, and if it is, add to the feature counter
                    if ( self.distanceFn.distance( vector1, vector2 ) <= self.radius ):
                        if featureName in rowDict:
                            if self.weights:
                                rowDict[featureName] += colParsed[-1]    # weighted version, colParsed[-1] is the weight
                            else:
                                rowDict[featureName] += 1    # unweighted version
                        else:
                            if self.weights:
                                rowDict[featureName] = colParsed[-1]     # weighted version, colParsed[-1] is the weight
                            else:
                                rowDict[featureName] = 1     # unweighted version
                            if featureName not in header: header.append(featureName)
                
                elif len(colParsed) > 1:
                    print('Error in secondary.csv file: location vector does not have enough terms')
                    
            xdata.append(rowDict)     # add the row of additional terms (encoded as a dictionary) to xdata
             
        # fuse xdata into primary data 
        self.data2 = Data()
        self.data2.processRawData( xdata, [], header, [ 'int' for i in range( len(header) ) ] )
        self.data1.addCols( self.data2 )
            
    def __str__(self):
        return(
        """Data fusing object: 
        distance function: %s
        primary data object:  %s
        """ % ( self.distanceFn, self.data1 )
        )

#########
if __name__ == '__main__':
    g = FuseData(Data(),"Euclidean")
    print ( g )

