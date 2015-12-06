import random
import cPickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

NUM_K_VALUES = 10
SPACING = 2
KMEANS_PICKLE_FNAME = "kmeans_results.pickle"

def runkmeans_sklearn(examples):
    d = DictVectorizer()
    X = d.fit_transform(examples)
    print "sparse matrix created"
    results = {}
    for k in range(1,NUM_K_VALUES + 1):
        print "running kmeans with K = ", SPACING*k
        kmeans = KMeans(SPACING*k, init='k-means++', n_init=5, verbose=True)
        km = kmeans.fit(X)
        results[SPACING*k] = km
    cPickle.dump(results, open(KMEANS_PICKLE_FNAME, 'wb'))
    return results


