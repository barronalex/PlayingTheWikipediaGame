import ucsUtil
import kmeans

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

MEDIA_WIKI_PREFIX = '{http://www.mediawiki.org/xml/export-0.10/}'
REDIRECT_STR1 = '#REDIRECT'
REDIRECT_STR2 = '#redirect'

GOAL_ARTICLE = 'Stanford'


stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')


# get the tree from the XML file
def init_pages():
    print 'starting'
    if os.path.exists(PICKLE_FNAME):
        print 'loading from pickle'
        pages = cPickle.load(open(PICKLE_FNAME, 'rb'))
    else:
        print 'loading from xml'
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
        setUpLinksAndFeatures()

    print 'wikipedia loaded'

    return pages

def setUpLinksAndFeatures():
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

    features = {token: 1 for token in unique_tokens_no_sw}
    features['NUM_WORDS'] = len(tokens)
    features['NUM_LINKS'] = len(links)

    return features


# set up pages dictionary which contains {page title: (page text, links from page, feature dict)}
pages = init_pages()
print 'pages is setup'



def perform_ucs():
    total_states_explored = 0
    total_cost = 0
    num_pith_paths = 0
    total_time = 0

    for i in range(0, 100):
        start_article = random.sample(pages, 1)[0]
        print 'start article: ', start_article
        print 'goal article: ', GOAL_ARTICLE
        start_time = datetime.datetime.now()
        search_prob = ucsUtil.SearchProblem(pages, start_article, GOAL_ARTICLE)
        ucs = ucsUtil.UniformCostSearch(1)
        ucs.solve(search_prob)
        end_time = datetime.datetime.now()
        total_time += int((end_time - start_time).microseconds)
        print ucs.actions
        total_states_explored += ucs.numStatesExplored
        if ucs.totalCost is None:
            continue
        total_cost += ucs.totalCost
        num_pith_paths += 1

        print ucs.totalCost
        print ucs.numStatesExplored
        print ''

    print 'av states explored: ', float(total_states_explored)/100
    print 'av cost: ', float(total_cost)/num_pith_paths
    print 'percent with paths: ', float(num_pith_paths), '%'
    print 'av time: ', float(total_time)/100
examples = [pages[page][2] for page in pages]
count = 0
assignments = []
for page in pages:
    assignments.append((page,count))
    count += 1
kmeans.runkmeans(examples,10, 300)
