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

import numpy
import matplotlib.pyplot as plot

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


WIKIPEDIA_XML_FNAME = 'simplewiki-latest-pages-articles.xml'
PICKLE_FNAME = 'pages.pickle'
TRAINING_DATA_PICKLE_FNAME = 'training_data.pickle'

MEDIA_WIKI_PREFIX = '{http://www.mediawiki.org/xml/export-0.10/}'
REDIRECT_STR1 = '#REDIRECT'
REDIRECT_STR2 = '#redirect'

INFINITE_COST = 10000

GOAL_ARTICLE = 'Stanford'


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
        print 'dumping to pickle'
        cPickle.dump(pages, open(PICKLE_FNAME, 'wb'))
        set_up_links_and_features()

    print 'wikipedia loaded'

    return pages


def set_up_links_and_features():
    print 'total articles:', len(pages)
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


# set up pages dictionary which contains {page title: (page text, links from page, feature dict)}
pages = init_pages()
print 'pages is setup'


def test_ucs(num_tests):
    total_states_explored = 0
    total_cost = 0
    num_pith_paths = 0
    total_time = 0

    for i in range(num_tests):
        start_article = random.sample(pages, 1)[0]
        print 'start article: ', start_article
        print 'goal article: ', GOAL_ARTICLE
        start_time = datetime.datetime.now()
        search_prob = ucs.SearchProblem(pages, start_article, GOAL_ARTICLE)
        ucs_prob = ucs.UniformCostSearch(1)
        ucs_prob.solve(search_prob)
        end_time = datetime.datetime.now()
        total_time += int((end_time - start_time).microseconds)
        print ucs_prob.actions
        total_states_explored += ucs_prob.numStatesExplored
        if ucs_prob.totalCost is None:
            continue
        total_cost += ucs_prob.totalCost
        num_pith_paths += 1

        print ucs_prob.totalCost
        print ucs_prob.numStatesExplored
        print ''

    print 'av states explored: ', float(total_states_explored)/100
    print 'av cost: ', float(total_cost)/num_pith_paths
    print 'percent with paths: ', float(num_pith_paths), '%'
    print 'av time: ', float(total_time)/100


def train_model(num_training_examples):
    print 'generating training data'

    if os.path.exists(TRAINING_DATA_PICKLE_FNAME):
        print 'loading from pickle'
        training_data = cPickle.load(open(TRAINING_DATA_PICKLE_FNAME, 'rb'))
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
            for j, action in enumerate(ucs_prob.actions):
                training_data[action] = num_actions - j

        print 'pickling for future use'
        cPickle.dump(pages, open(TRAINING_DATA_PICKLE_FNAME, 'wb'))

    x = []
    y = []
    for key, val in training_data.iteritems():
        x.append(pages[key][2])
        y.append(val)

    return classification.get_logistic_regression_model(x, y)


def test_model(num_testing_examples, model):
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
        for j, action in enumerate(ucs_prob.actions):
            training_data[action] = num_actions - j

    x = []
    y = []
    for key, val in training_data.iteritems():
        x.append(pages[key][2])
        y.append(val)

    print 'applying model'
    correct_classifications = classification.apply_logistic_regression_model(x, model)
    count = float(0)
    for i in range(len(y)):
        if y[i] == correct_classifications[i]:
            count += 1

    print count / num_testing_examples, '%'
    print count, 'of', num_testing_examples
    return count / num_testing_examples


def cluster_data(examples):
    kmeansresults = kmeans.runkmeans_sklearn(examples)


model = train_model(500)
test_model(100, model)

examples = [pages[page][2] for page in pages]
cluster_data(examples)


