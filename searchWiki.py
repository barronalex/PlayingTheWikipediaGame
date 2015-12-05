import re
import xml.etree.cElementTree as etree
import ucsUtil
import random
import datetime
import nltk
import numpy
import matplotlib.pyplot as plot


WIKIPEDIA_XML_FNAME = 'simplewiki-latest-pages-articles.xml'
MEDIA_WIKI_PREFIX = '{http://www.mediawiki.org/xml/export-0.10/}'
REDIRECT_STR1 = '#REDIRECT'
REDIRECT_STR2 = '#redirect'


GOAL_ARTICLE = 'Stanford'


# get the tree from the XML file
def init_tree():
    print 'starting'
    tree = etree.parse(WIKIPEDIA_XML_FNAME)
    print 'wikipedia loaded'
    root = tree.getroot()

    root = root[1:len(root)]
    pages = {}
    for page in root:
        title = page.find(MEDIA_WIKI_PREFIX + 'title')
        revision = page.find(MEDIA_WIKI_PREFIX + 'revision')
        text = revision.find(MEDIA_WIKI_PREFIX + 'text')
        if text is not None and title is not None:
            pages[title.text] = (text.text, None)

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


def extract_features(text):
    return nltk.word_tokenize(text)


# convert the page array to a tuple (article text, links, features)
pages = init_tree()
print 'total articles:', len(pages)
for i, (page, value) in enumerate(pages.iteritems()):
    val = value[0]
    links = get_links_from_text(pages, val)
    if not isinstance(val, basestring):
        print 'err', val
        features = []
    else:
        features = extract_features(val)
        if i % 1000 == 0:
            print i
    pages[page] = (val, links, features)
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

