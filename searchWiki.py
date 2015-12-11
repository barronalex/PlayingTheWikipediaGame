import ucs
import classification
import kmeans
import pca

import random
import datetime
import re
import os.path

import xml.etree.cElementTree as etree
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


WIKIPEDIA_XML_FNAME = 'simplewiki-latest-pages-articles.xml'
PICKLE_FNAME = 'pages.pickle'
TRAINING_DATA_PICKLE_FNAME = 'training_data.pickle'
MODEL_PICKLE_FNAME = 'model.pickle'

MEDIA_WIKI_PREFIX = '{http://www.mediawiki.org/xml/export-0.10/}'
REDIRECT_STR1 = '#REDIRECT'
REDIRECT_STR2 = '#redirect'

GOAL_ARTICLE = 'Stanford'
NUM_CLUSTERS = 4
INFINITE_COST = 10000


stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')


# get the tree from the XML file
def init_pages():
    print 'starting'
    if os.path.exists(PICKLE_FNAME):
        print 'loading pages from pickle'
        pages = cPickle.load(open(PICKLE_FNAME, 'rb'))
    else:
        print 'loading pages from xml'
        tree = etree.parse(WIKIPEDIA_XML_FNAME)
        root = tree.getroot()
        root = root[1:len(root)]
        pages = {}
        for page in root:
            title = page.find(MEDIA_WIKI_PREFIX + 'title')
            revision = page.find(MEDIA_WIKI_PREFIX + 'revision')
            text = revision.find(MEDIA_WIKI_PREFIX + 'text')
            if text is not None and title is not None:
                pages[title.text] = (text.text, None)
        print 'setting up links and pages'
        set_up_links_and_features(pages)
        print 'dumping to pickle'
        cPickle.dump(pages, open(PICKLE_FNAME, 'wb'))

    print 'wikipedia loaded'

    return pages
def get_links_from_text(pages, text):
    links = []
    if text is None:
        return []
    potential_links = re.findall(r"\[\[([A-Za-z0-9 _]+)\]\]", text)
    for potential_link in potential_links:
        if potential_link not in pages:
            continue
        link_text = pages[potential_link]
        if len(link_text) >= len(REDIRECT_STR1) and (link_text[0:len(REDIRECT_STR1)] == REDIRECT_STR1 or
                                            pages[potential_link][0:len(REDIRECT_STR1)] == REDIRECT_STR2):
            redir = re.findall(r"\[\[([A-Za-z0-9 _]+)\]\]", link_text)
            if len(redir) > 0:
                potential_link = redir[0]
            else:
                continue
        links.append(potential_link)
    return links
def extract_features(text, links):
    tokens = tokenizer.tokenize(text.lower())

    unique_tokens = set([])
    for token in tokens:
        unique_tokens.add(stemmer.stem(token))

    unique_tokens_no_sw = unique_tokens - stop_words

    features = {token: float(1) for token in unique_tokens_no_sw}
    features['NUM_WORDS'] = len(tokens)
    features['NUM_LINKS'] = len(links)

    return features
def set_up_links_and_features(pages):
    print 'total articles:', len(pages)
    ref = {}
    examples = []
    for i, (page, value) in enumerate(pages.iteritems()):
        val = value[0]
        links = get_links_from_text(pages, val)
        if not isinstance(val, basestring):
            features = {}
        else:
            features = extract_features(val, links)
            if i % 1000 == 0:
                print 'i = ', i
        pages[page] = (val, links, features)
        ref[page] = i
        examples.append(features)

    kmeans_results = kmeans.runkmeans_sklearn(examples, [NUM_CLUSTERS])
    labs = kmeans_results[NUM_CLUSTERS].labels_
    for page, value in pages.iteritems():
        features = value[2]
        cluster = labs[ref[page]]
        for i in range(NUM_CLUSTERS):
            features['IN_CLUSTER_'+str(i)] = 1 if cluster == i else 0


# set up pages dictionary which contains {page title: (page text, links from page, feature dict)}
pages = {}
pages = init_pages()
print 'pages is setup'
g, w, pv = ucs.make_Graph(pages)

def greedy_search(pages):
    model = train_models(100, ['heuristic'], [classification.get_logistic_regression_model_liblinear])[0]
    total_states_explored = 0
    num_with_paths = 0
    total_time = 0
    for i in range(100):
        start_time = datetime.datetime.now()
        current_article = random.sample(pages, 1)[0]
        print 'start article: ', current_article
        print 'goal article: ', GOAL_ARTICLE
        for j in range(10):
            links = pages[current_article][1]
            min_cost = 1000000
            for link in links:
                cost = classification.apply_model([pages[link][2]], model)
                if cost < min_cost:
                    print 'current min cost: ', min_cost
                    min_cost = cost
                    current_article = link
            if current_article == GOAL_ARTICLE:
                break
        end_time = datetime.datetime.now()
        total_states_explored += j
        print 'dist: ',j 
        if j == 9:
            continue 
        num_with_paths += 1

        total_time += int((end_time - start_time).microseconds)

    print 'av states explored:', float(total_states_explored)/(i+1)
    print 'percent with paths:', 100*float(num_with_paths)/(i+1), '%'
    print 'av time:', float(total_time)/(i+1)

def run_ucs(num_tests,g,w,pv, heuristic=None):
    total_states_explored = 0
    total_cost = 0
    num_with_paths = 0
    total_time = 0
    model = train_models(100, ['heuristic'], [classification.get_logistic_regression_model_liblinear])[0]
    for i in range(num_tests):
        start_time = datetime.datetime.now()
        start_article = random.sample(pages, 1)[0]
        print 'start article: ', start_article
        print 'goal article: ', GOAL_ARTICLE
        def h(v):
            guess = 1000000*classification.apply_model([pages[pv[v]][2]], model)
            #print guess
            return guess
        dist, pred, numExplored = ucs.search_Graph(start_article, g, w, pv, h) 
        end_time = datetime.datetime.now()
        total_states_explored += numExplored
        print dist
        if dist > 10:
            continue
        total_cost += dist 
        num_with_paths += 1

        total_time += int((end_time - start_time).microseconds)

    print 'av states explored:', float(total_states_explored)/(i+1)
    print 'av cost:', float(total_cost)/num_with_paths
    print 'percent with paths:', 100*float(num_with_paths)/(i+1), '%'
    print 'av time:', float(total_time)/(i+1)
    print ''

    return float(total_states_explored)/num_tests, float(total_cost)/num_with_paths, float(num_with_paths), \
        float(total_time)/num_tests


def train_models(num_training_examples, method_names, methods):
    print 'generating training data'

    fname = str(num_training_examples) + '.' + TRAINING_DATA_PICKLE_FNAME
    if os.path.exists(fname):
        print 'loading from pickle'
        training_data = cPickle.load(open(fname, 'rb'))
    else:
        print 'generating with UCS'
        training_data = {}

        for i in range(num_training_examples):
            if i % 10 == 0:
                print 'generated', i, 'examples'
            start_article = random.sample(pages, 1)[0]
            search_prob = ucs.SearchProblem(pages, start_article, GOAL_ARTICLE)
            ucs_prob = ucs.UniformCostSearch()
            ucs_prob.solve(search_prob)

            if ucs_prob.totalCost is None:
                training_data[start_article] = INFINITE_COST
                continue
            num_actions = len(ucs_prob.actions)
            training_data[start_article] = num_actions
            for j, action in enumerate(ucs_prob.actions):
                training_data[action] = num_actions - j - 1

        print 'pickling for future use'
        cPickle.dump(training_data, open(fname, 'wb'))

    x = []
    y = []
    for key, val in training_data.iteritems():
        x.append(pages[key][2])
        y.append(val)

    result = []
    for i, method in enumerate(methods):
        fname = method_names[i].lower().replace(' ', '_') + '.' + MODEL_PICKLE_FNAME
        if os.path.exists(fname):
            print 'loading model from pickle'
            result.append(cPickle.load(open(fname, 'rb')))
        else:
            mod = method(x, y)
            result.append(mod)
            print 'pickling for future use'
            cPickle.dump(mod, open(fname, 'wb'))

    return result
def test_models(num_testing_examples, models):
    print 'generating testing data'

    training_data = {}
    for i in range(num_testing_examples):
        if i % 10 == 0:
            print 'generated', i, 'examples'
        start_article = random.sample(pages, 1)[0]
        search_prob = ucs.SearchProblem(pages, start_article, GOAL_ARTICLE)
        ucs_prob = ucs.UniformCostSearch()
        ucs_prob.solve(search_prob)

        if ucs_prob.totalCost is None:
            training_data[start_article] = INFINITE_COST
            continue
        num_actions = len(ucs_prob.actions)
        training_data[start_article] = num_actions

    x = []
    y = []
    for key, val in training_data.iteritems():
        x.append(pages[key][2])
        y.append(val)

    results = {}
    for i, model in enumerate(models):
        print 'applying model', i
        classifications = classification.apply_model(x, model)
        correct_count = 0
        reachable_count = 0
        wrong_inf_count = 0
        dist = 0
        for j in range(len(y)):
            if y[j] == INFINITE_COST:
                continue
            reachable_count += 1
            if y[j] == classifications[j]:
                correct_count += 1
            else:
                if y[j] == INFINITE_COST or classifications[j] == INFINITE_COST:
                    wrong_inf_count += 1
                else:
                    dist += abs(y[j] - classifications[j])

        results[model] = (correct_count, dist, wrong_inf_count, reachable_count)
        print type(model)
        print 'fully correct', 100 * float(correct_count) / reachable_count, '%'
        print 'dist:', float(dist) / reachable_count
        print 'wrong inf.s:', 100 * float(wrong_inf_count) / reachable_count, '%'
        print ''

    return results
def try_models():
    method_names = ['One vs. Rest Logistic Regression',
                    'Multinomial Logistic Regression',
                    'SGD with Hinge Loss',
                    'SGD with Percepton Loss',
                    'SVM with Squared Hinge Loss']
    methods = [classification.get_logistic_regression_model_liblinear,
               classification.get_logistic_regression_model_lbfgs_multinomial,
               classification.get_sgd_model_hinge,
               classification.get_sgd_model_perceptron,
               classification.get_svm_model]
    models = train_models(1000, method_names, methods)
    test_results = test_models(200, models)


def cluster_data():
    examples = [pages[page][2] for page in pages]
    kmeans_results = kmeans.runkmeans_sklearn(examples, range(1, 16))
    x = sorted(kmeans_results.keys())
    y = [kmeans_results[key].inertia_ for key in x]
    plt.plot(x, y)

    plt.xlabel('Number of Clusters')
    plt.ylabel('Loss')
    plt.title('K-means Clustering\nLoss vs. Number of Clusters')
    plt.savefig('kmeans_cluster_num_graph.png')
    plt.show()


def test_heuristic():
    model = train_models(2000, ['heuristic'], [classification.get_logistic_regression_model_liblinear])[0]

    def heuristic(link):
        return classification.apply_model([pages[link][2]], model)
    run_ucs(100, heuristic=heuristic)



