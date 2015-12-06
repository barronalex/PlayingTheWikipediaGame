import os.path
import cPickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

NUM_K_VALUES = 10
SPACING = 2
N_INITS = 5
KMEANS_PICKLE_FNAME = "kmeans_results.pickle"


def runkmeans_sklearn(examples):
    fname = str(NUM_K_VALUES)+'.'+str(SPACING)+'.'+KMEANS_PICKLE_FNAME
    if os.path.exists(fname):
        print 'loading from pickle'
        return cPickle.load(open(fname, 'rb'))

    d = DictVectorizer()
    X = d.fit_transform(examples)
    print "sparse matrix created"
    results = {}
    for k in range(1, NUM_K_VALUES + 1):
        print "running kmeans with K = ", SPACING*k
        kmeans = KMeans(SPACING*k, init='k-means++', n_init=N_INITS, verbose=True)
        km = kmeans.fit(X)
        results[SPACING*k] = km
    cPickle.dump(results, open(fname, 'wb'))
    return results


