#!/usr/bin/env python
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import RandomizedPCA, MiniBatchSparsePCA, FastICA
from sklearn.manifold import MDS, SpectralEmbedding
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class BaseC(object):
    def _plot(self, plt, X, title, labels, unique_labels, coremarkersize=6, outmarkersize=2):
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        plt.figure()
        plt.title(title)
        for k, col in zip(unique_labels, colors):
            markersize = coremarkersize
            alpha = 0.5
            if k == -1: # DBSCAN returns -1 for noise - plot those points using black.
                col = 'k'
                markersize = outmarkersize
                alpha = 0.1
            class_members = [index[0] for index in np.argwhere(labels == k)]
            for index in class_members:
                x = X[index]
                plt.plot(x[0], x[1], 'o', markerfacecolor=col, markeredgecolor='k',
                         alpha = alpha, markersize=markersize)       
        
    def plot(self, X, title="Clustering", pdffile='clusters.pdf', show=True):
        unique_labels = set(self.labels_)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        X = StandardScaler().fit_transform(X)
        with PdfPages(pdffile) as pdf:
            fig = plt.figure()
            fig.text(0.3, 0.6, title, fontsize=30, fontweight='bold')
            fig.text(0.25, 0.5, 'Estimated number of clusters: %d' % num_clusters, style='italic',
                    bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            fig.text(0.2, 0.4, 'This file contains 2D projections of the clustered data')
            fig.text(0.2, 0.35, 'Cluster colors remain same between plots')
            fig.text(0.2, 0.3, 'Suspected outliers are plotted in black and with smaller dots')
            pdf.savefig()
            grp = GaussianRandomProjection(n_components=2)
            Xred = grp.fit_transform(X)
            self._plot(plt, Xred, 'Random projection to 2D', self.labels_, unique_labels)
            pdf.savefig()
            pca = RandomizedPCA(n_components=2)
            Xred = pca.fit_transform(X)
            self._plot(plt, Xred, 'First two principal components', self.labels_, unique_labels)
            pdf.savefig()
            ica = FastICA(n_components=2)
            Xred = ica.fit_transform(X)
            self._plot(plt, Xred, 'First two independent components', self.labels_, unique_labels)
            pdf.savefig()
            mds = MDS(n_components=2, metric=True)
            Xred = mds.fit_transform(X)
            self._plot(plt, Xred, 'Multidimensional scaling', self.labels_, unique_labels)
            pdf.savefig()
            sp = SpectralEmbedding(n_components=2)
            Xred = sp.fit_transform(X)
            self._plot(plt, Xred, 'Spectral embedding', self.labels_, unique_labels)
            pdf.savefig()
            if show:
                plt.show()
        
    def score(self, X, labels_true=None):
        # Both these metrics (silhouette score and the adjusted rand score range between [-1,1]. To
        # keep things consistent with other metrics like accuracy, R^2, precision, recall etc., we
        # scale the silhouette and rand score so that they lie between [0,1].
        if labels_true is None:
            return (1.0 + metrics.silhouette_score(X, self.labels_)) / 2.0
        else:
            return (1.0 + metrics.adjusted_rand_score(labels_true, self.labels_)) / 2.0   
        
class DbscanC(BaseC, DBSCAN):
    def fit(self, X, dummy=None):  # Dummy variable enables use in grid/random search
        super(DbscanC, self).fit(StandardScaler().fit_transform(X))

    def clone(self):
        return DbscanC()
        
class KmeansC(BaseC, MiniBatchKMeans):
    def fit(self, X, dummy=None):  # Dummy variable enables use in grid/random search
        super(KmeansC, self).fit(StandardScaler().fit_transform(X))
        
    def clone(self):
        return KmeansC()
        

######################
if __name__ == '__main__':
    from Data import Data
    doorA = Data('JRE Door A Current')
    doorA.readRawDataFromCSV("jr_door_current.csv", hasY=False)
    db = DbscanC(min_samples=20)
    db.fit(doorA.X)
    print ( "Silhouette:", db.score(doorA.X) )
    db.plot(doorA.X, 'JR DoorA')    
    #from sklearn.datasets.samples_generator import make_blobs
    #centers = [[5, 5], [-2, -1], [3, -1]]
    #X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=1.0, random_state=0)
    #db = DbscanC(eps=0.3, min_samples=10)
    #db.fit(X)
    #print "Silhouette:", db.score(X)
    #print "AR:", db.score(X, labels_true)
    #db.plot(X, db.labels_)
    #km = KmeansC(n_clusters=3)
    #km.fit(X)
    #print "Silhouette:", km.score(X)
    #print "AR:", km.score(X, labels_true)    