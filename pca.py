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
import re

PCA_PICKLE_FNAME = 'pca_components.pickle'
GOAL_ARTICLE = 'Stanford'
NUM_EXAMPLE_PATHS = 100
EXAMPLE_DENSITIES_FNAME = 'example_densities.pickle'
NUM_COMPONENTS = 3

def run_pca(examples):
    svd = TruncatedSVD(n_components=NUM_COMPONENTS, random_state=42)
    d = DictVectorizer()
    X = d.fit_transform(examples)
    X = csr_matrix.transpose(X)
    print "sparse matrix obtained"
    components = svd.fit(X).components_
    cPickle.dump(svd.components_, open(PCA_PICKLE_FNAME, 'wb'))
    return components

    
def graph_pca(examples, pages):
    if os.path.exists(PCA_PICKLE_FNAME):
        print 'loading from pickle'
        components = cPickle.load(open(PCA_PICKLE_FNAME, 'rb'))
    else:
        components = run_pca(examples)
    pageIndices = {}
    for j, page in enumerate(pages):
        pageIndices[page] = j
    numTimesStateVisited = {}
    paths = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    start_articles = []
    for i in range(1000):
        if i % 10 == 0:
            print 'generated', i, 'examples'
        start_article = random.sample(pages, 1)[0]
        start_articles.append(start_article)
        startState = components[:,pageIndices[start_article]]
        ax.scatter([startState[0]],[startState[1]],[startState[2]],s=50,c='g')
        search_prob = ucs.SearchProblem(pages, start_article, GOAL_ARTICLE)
        ucs_prob = ucs.UniformCostSearch()
        ucs_prob.solve(search_prob)
        pathReducedStates = [components[:,pageIndices[start_article]]]
        if ucs_prob.actions is None: continue
        for action in ucs_prob.actions:
            pathReducedStates.append(components[:,pageIndices[action]])
            if action == 'Stanford' or action == start_article: continue 
            if action not in numTimesStateVisited:
                numTimesStateVisited[action] = 1
            else:
                numTimesStateVisited[action] += 1
        paths.append(numpy.array(pathReducedStates))
    points = []
    st = []
    for state, density in numTimesStateVisited.iteritems():
        points.append(components[:,pageIndices[state]])
        st.append(20 + density**(1.5))
        print density 
    pointsNP = numpy.array(points)
    densities = (points,st,start_articles)
    cPickle.dump(densities, open(EXAMPLE_DENSITIES_FNAME, 'wb'))
    x = pointsNP[:,0]
    y = pointsNP[:,1]
    z = pointsNP[:,2]
    print st
    goalState = components[:,pageIndices[GOAL_ARTICLE]]
    ax.scatter([goalState[0]],[goalState[1]],[goalState[2]],s=200,c='r')
    ax.scatter(x,y,z,s=st,c='b',depthshade=False)
    plt.show(block = False)

    return numTimesStateVisited

def graph_entire_2D_pca(components, pages):
    x = components[0,:]
    y = components[1,:]
    dates = []
    for i,state in enumerate(pages):
        if re.match(r"[0-90-90-90-9]", state) is not None:
            dates.append(i)
        if re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)", state) is not None:
            dates.append(i)
        if re.search(r"(List)", state) is not None:
            print i
            dates.append(i)
    for index in dates:
        plt.scatter([components[0,index]],[components[1,index]],s=200,c='r')
    plt.scatter(x,y)
    plt.xlabel("pca component 1")
    plt.ylabel("pca component 2")
    plt.title("2D pca with Dates Highlighted")
    plt.show(block = False)
    
def graph_entire_3D_pca(components, pages):
    dates = []
    x = components[0,:]
    y = components[1,:]
    z = components[2,:]
    for j in range(0,len(x)):
        if random.random() < 0.5:
            x[j] *= -1
        if random.random() < 0.5:
            y[j] *= -1
        if random.random() < 0.5:
            z[j] *= -1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.axis('off')
    plt.show(block = False)


