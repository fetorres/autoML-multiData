# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:37:24 2016

@author: torres
"""
#import logging, logging.config
import math
import numpy

class DistanceFn(object):
    def __init__(self, method, logConf='logging.conf'):
        self.method = method
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')

    def distance(self, vector1, vector2):    
        ''' in case one of the vectors is padded with extra terms, don't use 
            those terms
        '''
        vlength = min( len(vector1), len(vector2) )
        self.vector1 = vector1[0:vlength]
        self.vector2 = vector2[0:vlength]
        return getattr(self, self.method )()
        
    def L_1Norm(self):
        vdiff = numpy.array(self.vector1) - numpy.array(self.vector2)
        return numpy.linalg.norm(vdiff, ord=1 )
            
    def euclidean(self):
        vdiff = numpy.array(self.vector1) - numpy.array(self.vector2)        
        return numpy.linalg.norm(vdiff, ord=2 )
            
    def L_infinityNorm(self):
        vdiff = numpy.array(self.vector1) - numpy.array(self.vector2)        
        return numpy.linalg.norm(vdiff, numpy.inf )
            
    def distanceOnEarth(self):
        '''  Assumes the vectors are [latitude, longitude], and Earth is a 
        sphere with radius 6.371e6 meters.
        Returns distance in meters.
        '''
        # Converts lat & long to radians.
        # lat ~ phi
        # long ~ theta
        degrees_to_radians = math.pi/180.0
        phi1 = self.vector1[0]*degrees_to_radians
        phi2 = self.vector2[0]*degrees_to_radians
        theta1 = self.vector1[1]*degrees_to_radians
        theta2 = self.vector2[1]*degrees_to_radians
        # Compute the spherical distance from spherical coordinates using haversine formula 
        # For two locations in spherical coordinates:
        # distance = radius * dsigma
        # dsigma = 2 asin( sqrt( arg) )  where
        # arg = sin^2( (lat1-lat2)/2 ) + cos(lat1) * cos(lat2) * sin^2( (long1-long2)/2 )
        sinsq = ( ( math.sin(0.5 * (phi1-phi2)) )**2 
              + math.cos(phi1)*math.cos(phi2)*( math.sin( 0.5 * (theta1-theta2) ) )**2 )
        arc = 2 * math.asin(math.sqrt(sinsq)) * 6.371e6 #radius of the earth in m
        return arc
        
    def L_1Norm_cat(self):
        '''  For categorical variables.  For each element, difference is 0 when 
        they are the same, 1 when different.
        '''
        temp = [ int( c1!=c2 ) for c1, c2 in zip( self.vector1, self.vector2 ) ]
        return sum( temp ) / len( temp )
            
    def L_infinityNorm_cat(self):
        '''  For categorical variables.  For each element, difference is 0 when 
        they are the same, 1 when different.
        '''
        temp = [ int( c1!=c2 ) for c1, c2 in zip( self.vector1, self.vector2 ) ]
        return max( temp )
            
    def __str__(self):
        return(
        """Distance object
        method for calculating distance: %s
        """ % ( self.method )
        )

#########
if __name__ == '__main__':
    g = DistanceFn("Euclidean")
    print ( g )

