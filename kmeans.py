import os.path
import cPickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans


KMEANS_PICKLE_FNAME = "kmeans_results.pickle"


def runkmeans_sklearn(examples, cluster_nums):
    d = DictVectorizer()
    X = d.fit_transform(examples)
    print "sparse matrix created"

    result = {}
    for cluster_num in cluster_nums:
        fname = str(cluster_num)+'.'+KMEANS_PICKLE_FNAME
        if os.path.exists(fname):
            print 'loading from pickle K =', cluster_num
            result[cluster_num] = cPickle.load(open(fname, 'rb'))
            continue

        print 'running kmeans with K = ', cluster_num
        kmeans = KMeans(cluster_num, init='k-means++', n_init=5, verbose=True)
        km = kmeans.fit(X)
        result[cluster_num] = km
        print 'pickling'
        cPickle.dump(km, open(fname, 'wb'))

    return result

