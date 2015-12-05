import random
def runkmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
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
