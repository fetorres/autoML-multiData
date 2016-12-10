#!/usr/bin/env python
from scipy.misc import imread, imresize, imsave, imshow
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pylab as pl
import logging, logging.config

class Image(object):
    """Process image objects and provide data in a form usable by machine learning algorithms."""
    def __init__(self, name='', img=None, logConf='logging.conf'):
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.name = name
        self.filename = None
        self.img = img
        
    def readFromFile(self, filename):
        """Read an image from the specified filename. The image is converted to monochrome."""
        #self.logger.debug('Reading image: %s' % filename)
        print('Reading image: %s' % filename)
        self.filename = filename
        self.img = imread(filename, flatten=True)  # Monochrome images
        if not self.name:
            self.name = filename

    def saveToFile(self, filename):
        """Write the image to the specified filename."""
        imsave(filename, self.img)
        
    def getClusters(self, n_clusters=10, size=32.):
        """Return the specified number of clusters - each cluster is defined based on gradient boundaries. In order
        to speed up the analysis, the image is downsampled prior to the clustering analysis. The maximum side of the
        downsampled image is specified by the size parameter."""
        imgsize = max(self.img.shape)
        if imgsize > size:
            resizeFraction = size / imgsize
        else:
            resizeFraction = 1.
        img = imresize(self.img, resizeFraction)
        # Convert the image into a graph with the value of the gradient on the edges.
        graph = image.img_to_graph(img)  
        # Take a decreasing function of the gradient: an exponential
        # The smaller beta is, the more independent the segmentation is of the
        # actual image. For beta=1, the segmentation is close to a voronoi
        beta = 5
        eps = 1e-6
        graph.data = np.exp(-beta * graph.data / img.std()) + eps
        try:
            clusters = spectral_clustering(graph, n_clusters=n_clusters, assign_labels='discretize')
        except ( Exception, msg ) :
            #self.logger.warn("Error computing clusters on %s: %s. Returning a single cluster." % (self.filename, msg))
            print("Error computing clusters on %s: %s. Returning a single cluster." % (self.filename, msg))
            imgsize = img.shape[0] * img.shape[1]
            clusters = [0]*(imgsize-n_clusters+1)
            clusters.extend(range(1,n_clusters))
        return (img, clusters)
    
    def getFeatures(self, n_features=10, size=32., normalize=True, includePixels=True):
        """Return the features extracted from the image. Currently, it returns cluster means and (optionally) the
        pixel values. The 'normalize' parameter specifies whether the feature values should be normalized to lie
        between 0 and 1. The image is downsampled prior to analysis - the maximum number of pixels on the longer
        side is specified by the 'size' parameter."""
        img, clusterLabels = self.getClusters(n_features, size)
        clusterLabels = np.array(clusterLabels)
        labelOrder = []
        labelSet = set()
        # Order the cluster labels in the order of appearance in the image - hopefully this will results
        # in consistent ordering across a batch of images.
        for cl in clusterLabels:
            if cl not in labelSet:
                labelOrder.append(cl)
                labelSet.add(cl)
        features = np.array([img.flatten()[clusterLabels == i].mean() for i in labelOrder])  # cluster means
        if includePixels:   # if includePixels is True then add those values to the cluster means
            features = np.hstack((features, img.flatten()))
        if normalize:
            return features / 256.
        else:
            return features
    
    def getValues(self, size=32.):
        """Returns the pixel values as a vector. The image is downsampled to have no more than 'size' number of pixels
        on the longer side."""
        imgsize = max(self.img.shape)
        if imgsize > size:
            resizeFraction = size / imgsize
        else:
            resizeFraction = 1.
        return imresize(self.img, resizeFraction).flatten()
    
    def show(self):
        """Display the image."""
        pl.figure(figsize=(5, 5))
        pl.imshow(self.img, cmap=pl.cm.gray)
        pl.xticks(())
        pl.yticks(())
        pl.title(self.name)
        pl.show()
    
    def __str__(self):
        """Return the image dimensions if it is non-empty."""
        if self.img is not None:
            return "Image size: %d x %d" % (self.img.shape[0], self.img.shape[1])
        else:
            return "Emtpy image"

##########
if __name__ == '__main__':
    #i = Image('Traffic Camera')
    #i.readFromFile('trafficCameras/NT0100_2012080913052900_2.jpg')
    #i.show()
    j = Image('Elevator')
    j.readFromFile('elevator.jpg')
    j.show()
    jimg, clusters = j.getClusters()
    jc = Image('Elevator clusters', np.array(clusters).reshape(jimg.shape))
    jc.show()
    print ( j.getFeatures() )
    
    
    #from sklearn.datasets import load_digits
    #import csv
    #d = load_digits()
    #with open('digits.csv', 'wb') as f:
    #    cf = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #    cf.writerow(['x__imgfile','y__cat'])
    #    for i, imgdata in enumerate(d['images']):
    #        img = Image('digit_%d'%i, imgdata)
    #        dig = 'digits/digit_%d.jpg'%i
    #        img.saveToFile(dig)
    #        cf.writerow([dig, d['target'][i]])