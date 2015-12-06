import random
import sklearn
from sklearn.feature_extraction import DictVectorizer

def runkmeans_sklearn(examples, K):
    d = DictVectorizer()
    X = d.fit_transform(examples)
    kmeans = sklearn.cluster.KMeans(K, init='k-means++', n_init=1, verbose=True)
    kmeans.fit(X)


def runkmeans(examples, K, maxIters):
    def increment(d1, scale, d2):
        for f, v in d2.items():
            d1[f] = d1.get(f, 0) + v * scale
    def norm2(vec):
        norm = 0
        for key in vec:
            norm += vec[key]**2
        return norm
    def incrementRet(d1, scale, d2):
        d3 = {}
        for f,v in d1.items():
            d3[f] = d1[f]
        for f, v in d2.items():
            d3[f] = d1.get(f, 0) + v * scale
        return d3
    def sumVecs(vecs):
        result = {}
        for vec in vecs:
            scale = 1/float(len(vecs))
            increment(result, scale, vec)
        return result
    def loss():
        sumElems = [norm2(incrementRet(examples[i], -1, centroids[z[i]])) for i in range(0,len(examples))]
        loss = float(0)
        for x in sumElems:
            loss += x
        return loss
    centroids = random.sample(examples,K)
    z = [0 for x in range(0, len(examples))]
    for k in range(0,maxIters):
        print "iter: ", k
        for i in range(0,len(examples)):
            curEx = examples[i] 
            z[i] = centroids.index(min(centroids, key=lambda cen: norm2(incrementRet(curEx, -1, cen))))
        lastLoss = loss()
        print "E step"
        for i in range(0,len(centroids)):
            pointsInCluster = [examples[j] for j in range(0,len(z)) if z[j] == i]
            if len(pointsInCluster) > 0:
                centroids[i] = sumVecs(pointsInCluster)
        print "M step"
        
    return (centroids, z)
