from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import ucs 
import os.path
import cPickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy

PCA_PICKLE_FNAME = 'pca_components.pickle'
GOAL_ARTICLE = 'Stanford'

def run_pca(examples):
    svd = TruncatedSVD(n_components=3, random_state=42)
    d = DictVectorizer()
    X = d.fit_transform(examples)
    X = csr_matrix.transpose(X)
    print "sparse matrix obtained"
    cPickle.dump(svd.components_, open(PCA_PICKLE_FNAME, 'wb'))
    return svd.fit(X).components_

def graph_pca(examples, pages):
    if os.path.exists(PCA_PICKLE_FNAME):
        print 'loading from pickle'
        components = cPickle.load(open(PCA_PICKLE_FNAME, 'rb'))
    else:
        components = run_pca(examples)
    count = 0
    pageIndices = {}
    for page in pages:
        pageIndices[page] = count
        count += 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,10):
        while(1):
            start_article = random.sample(pages, 1)[0]
            print start_article
            search_prob = ucs.SearchProblem(pages, start_article, GOAL_ARTICLE)
            ucs_prob = ucs.UniformCostSearch()
            ucs_prob.solve(search_prob)
            pathReducedStates = [components[:,pageIndices[start_article]]]
            actions = ucs_prob.actions
            print actions
            if actions is not None:
                for action in actions:
                    pathReducedStates.append(components[:,pageIndices[action]])    
                break
        pathReducedStatesNP = numpy.array(pathReducedStates)
        print pathReducedStatesNP
        x = pathReducedStatesNP[:,0]
        y = pathReducedStatesNP[:,1]
        z = pathReducedStatesNP[:,2]
        plt.plot(x,y,z)
    plt.show(block = False)
    

