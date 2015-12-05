import os.path
import re
import xml.etree.cElementTree as etree
import util
import random
import datetime


WIKIPEDIA_XML_FNAME = 'simplewiki-latest-pages-articles.xml'
print 'starting'
tree = etree.parse(WIKIPEDIA_XML_FNAME)
print 'tree obtained'
root = tree.getroot()
print root

def removePrefixFromTag(tag):
    return tag.rsplit('}', 1)[1]

MEDIA_WIKI_PREFIX = '{http://www.mediawiki.org/xml/export-0.10/}'
REDIRECT_STR1 = '#REDIRECT'
REDIRECT_STR2 = '#redirect'

root = root[1:len(root)]
pages = {}
for page in root:
    title = page.find(MEDIA_WIKI_PREFIX + 'title')
    revision = page.find(MEDIA_WIKI_PREFIX + 'revision')
    text = revision.find(MEDIA_WIKI_PREFIX + 'text')
    if text is not None and title is not None:
        pages[title.text] = (text.text, None)

def getLinksFromText(text):
    links = []
    if text is None: return [] 
    potentialLinks = re.findall(r"\[\[([A-Za-z0-9 _]+)\]\]",text)
    for potentialLink in potentialLinks:
        if potentialLink not in pages:
            continue
        linkText = pages[potentialLink]
        if len(linkText) >= len(REDIRECT_STR1) and (linkText[0:len(REDIRECT_STR1)] == REDIRECT_STR1 \
                or pages[potentialLink][0:len(REDIRECT_STR1)] == REDIRECT_STR2):
            redir = re.findall(r"\[\[([A-Za-z0-9 _]+)\]\]",linkText)
            if len(redir) > 0:
                potentialLink = redir[0]
            else: continue
        links.append(potentialLink)
    return links

for page, value in pages.iteritems():
    pages[page] = (value[0], getLinksFromText(value[0]))

totalStatesExplored = 0
totalCost = 0
numWithPaths = 0
totalTime = 0
for i in range(0,100):
    start_article = random.sample(pages, 1)[0]
    print 'start article: ', start_article
    goal_article = 'Stanford'
    startTime = datetime.datetime.now()
    search_prob = util.SearchProblem(pages, start_article, goal_article)
    ucs = util.UniformCostSearch(1)
    ucs.solve(search_prob)
    endTime = datetime.datetime.now()
    totalTime += int((endTime - startTime).microseconds)
    print ucs.actions
    totalStatesExplored += ucs.numStatesExplored
    if ucs.totalCost is None: continue
    totalCost += ucs.totalCost
    numWithPaths += 1

    print ucs.totalCost
    print ucs.numStatesExplored
print 'av states explored: ', float(totalStatesExplored)/100
print 'av cost: ', float(totalCost)/numWithPaths
print 'percent with paths: ', float(numWithPaths), '%'
print 'av time: ', float(totalTime)/100




