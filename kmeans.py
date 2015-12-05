import random
import sklearn


def runkmeans_sklearn(examples, K, maxIters):
    kmeans = sklearn.cluster.KMeans(8, init='k-means++', n_init=5)


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
        for i in range(0,len(examples)):
            curEx = examples[i] 
            z[i] = centroids.index(min(centroids, key=lambda cen: norm2(incrementRet(curEx, -1, cen))))
        lastLoss = loss()
        for i in range(0,len(centroids)):
            pointsInCluster = [examples[j] for j in range(0,len(z)) if z[j] == i]
            if len(pointsInCluster) > 0:
                centroids[i] = sumVecs(pointsInCluster)
        if loss() == lastLoss:
            break
    return (centroids, z, loss())
