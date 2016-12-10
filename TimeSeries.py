#!/usr/bin/env python
from pyearth import Earth
import numpy as np
import logging, logging.config

class TimeSeries(object):
    def __init__(self, ts, max_terms=100, penalty=3.0, logConf='logging.conf'):
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')
        self.y = np.array(ts)
        self.x = np.linspace(0., 1., num=len(ts))
        self.model = Earth(max_terms=max_terms, penalty=penalty)
        self.model.fit(self.x, self.y)
        #self.logger.debug(self.model.trace())
        #self.logger.debug(self.model.summary())

    def getFeatures(self):
        """Return global statistics (mean, P0, P25, P50, P75, P100) as well slopes of local segments."""
        b = [m for m in self.model.basis_ if not m.is_pruned()]
        knots = []
        for m in b:
            if m.has_knot():
                knots.append(m.get_knot())
            else:
                knots.append(0)
        return np.append([np.mean(self.y), np.min(self.y), np.percentile(self.y, 25), np.median(self.y), np.percentile(self.y, 75), np.max(self.y)],
                        [x for (y,x) in sorted(zip(knots, self.model.coef_))])


#############
if __name__ == '__main__':
    from matplotlib import pyplot

    #Create some fake data
    m = 1000
    n = 1
    X = np.linspace(0, 1, m)
    y = np.sin(10.*X) + np.cos(20.*X)
    ts = TimeSeries(y)
    print ( ts.getFeatures() )

    #Plot the model
    y_hat = ts.model.predict(X)
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('Simple Earth Example')
    pyplot.show()
