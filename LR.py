#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from datetime import datetime, date, time
from nltk.tokenize import sent_tokenize
from numpy import matrix
import csv
import json
import numpy as np
import re

TRAINING_PATH = '/yelp_training_set/'
TEST_PATH = '/yelp_test_set/'

class Review(object):
    def __init__(
        self,
        bizid,
        uid,
        stars,
        text,
        date,
        votes,
        revid
        ):
        self.bizid = bizid
        self.uid = uid
        self.stars = stars
        self.text = text
        self.date = datetime.strptime(date, '%Y-%m-%d')
        self.votes = votes
        self.revid = revid

class Business(object):
    def __init__(
        self,
        bizid,
        stars,
        categories,
        longitude,
        latitude,
        reviewCount,
        ):
        self.bizid = bizid
        self.stars = stars
        self.categories = categories
        self.longitude = longitude
        self.latitude = latitude
        self.reviewCount = reviewCount

class User(object):
    def __init__(
        self,
        uid,
        stars,
        reviewCount,
        votes,
        ):
        self.uid = uid
        self.stars = stars
        self.votes = votes
        self.reviewCount = reviewCount

    def VPR(self):
        return self.votes['useful'] / self.reviewCount

def lookUpBiz(bizid):
    if bizid in BizMap:
        return BizMap[bizid]
    return None

def lookUpUser(uid):
    if uid in UserMap:
        return UserMap[uid]
    return None

def read_file(file):
    lines = []
    with open(file) as f:
        for line in f:
            lines.append(json.loads(line))
    f.close()
    return lines

def read_training_file(file):
    return read_file(TRAINING_PATH + file)

def read_test_file(file):
    return read_file(TEST_PATH + file)

d = date(2013, 1, 19)
t = time(0, 0)
measuredDate = datetime.combine(d, t)


BizMap = {}
for bb in read_training_file('yelp_training_set_business.json'):
    b = Business(bb['business_id'], bb['stars'], bb['categories'],
        bb['longitude'], bb['latitude'], bb['review_count'])
    BizMap[b.bizid] = b

UserMap = {}
for bb in read_training_file('yelp_training_set_user.json'):
    b = User(bb['user_id'], bb['average_stars'], bb['review_count'],
             bb['votes'])
    UserMap[b.uid] = b

Reviews = []
for rev in read_training_file('yelp_training_set_review.json'):
    r = Review(rev['business_id'], rev['user_id'], rev['stars'],
        rev['text'], rev['date'], rev['votes'], None)
    Reviews.append(r)

vallim = [len(Reviews), 0, 0]
allInput = []
for rev in Reviews:
    thisBiz = lookUpBiz(rev.bizid)
    excount = 0
    for sent in sent_tokenize(rev.text):
        ss = sent.strip()
        if ss.endswith('!'):
            excount += 1
    if len(rev.text) == 0:
        allInput.append([
            thisBiz.stars,
            len(rev.text),
            rev.stars,
            abs(rev.date - measuredDate).days,
            0,
            0,
            0,
            0,
            0,
            thisBiz.longitude,
            thisBiz.latitude,
            ])
    else:
        allInput.append([
            thisBiz.stars,
            len(rev.text),
            rev.stars,
            abs(rev.date - measuredDate).days,
            excount,
            np.mean([len(sent) for sent in sent_tokenize(rev.text)]),
            len(sent_tokenize(rev.text)),
            len(re.findall('\n\n', rev.text)) + 1,
            len(rev.text.splitlines()[0]),
            thisBiz.longitude,
            thisBiz.latitude,
            ])

allUseful = [[rev.votes['useful']] for rev in Reviews]
MM = matrix(allInput)
bizLR = (MM.T * MM).I * MM.T
bizw = bizLR * matrix(allUseful)
bizwarray = [bizw[i, 0] for i in range(len(bizw))]


## biz stars, user rc, len(text), stars, days old

def inputfactx(rev, include_vpr):
    thisBiz = lookUpBiz(rev.bizid)
    thisUser = lookUpUser(rev.uid)
    result = [ thisBiz.stars ]
    if include_vpr:
        result += [ thisUser.VPR() ]
    result += [
        thisUser.reviewCount,
        len(rev.text),
        rev.stars,
        abs(rev.date - measuredDate).days ]
    if len(rev.text) == 0:
        result += [ 0, 0, 0, 0, 0 ]
    else:
        excount = 0
        for sent in sent_tokenize(rev.text):
            ss = sent.strip()
            if ss.endswith('!'):
                excount += 1
        result += [ excount,
        np.mean([len(sent) for sent in sent_tokenize(rev.text)]),
        len(sent_tokenize(rev.text)),
        len(re.findall('\n\n', rev.text)) + 1,
        len(rev.text.splitlines()[0]) ]
    result += [ thisBiz.longitude, thisBiz.latitude ]
    return result

def inputfact(rev):
    return inputfactx(rev, True)

def inputfact2(rev):
    return inputfactx(rev, False)

count = 0
tempval = 0
inputFactor3 = []
inputFactor2 = []
selectedRevList3 = []
selectedRevList2 = []
for rev in Reviews:
    if rev.votes['useful'] < 6 and abs(rev.date - measuredDate).days < 1800 and len(rev.text) < 1930:
        tempval += 1
        thisBiz = lookUpBiz(rev.bizid)
        thisUser = lookUpUser(rev.uid)
        if thisBiz is not None and thisUser is not None and thisUser.reviewCount < 513:
            selectedRevList2.append(rev)
            inputFactor2.append(inputfact2(rev))
            if thisUser.VPR() < 6:
                inputFactor3.append(inputfact(rev))
                selectedRevList3.append(rev)
    if count == vallim[0] - 1:
        vallim[2] = tempval
    count += 1

usefulList3 = [[rev.votes['useful']] for rev in selectedRevList3]
usefulList2 = [[rev.votes['useful']] for rev in selectedRevList2]

AA = matrix(inputFactor3)
LRmatrix = (AA.T * AA).I * AA.T
w = LRmatrix * matrix(usefulList3)
warray = [w[i, 0] for i in range(len(w))]

AA2 = matrix(inputFactor2)
LRmatrix2 = (AA2.T * AA2).I * AA2.T
w2 = LRmatrix2 * matrix(usefulList2)
warray2 = [w2[i, 0] for i in range(len(w2))]

##### test #####

for bb in read_test_file('yelp_test_set_business.json'):
    b = Business(
        bb['business_id'],
        bb['stars'],
        bb['categories'],
        bb['longitude'],
        bb['latitude'],
        bb['review_count'],
        )
    BizMap[b.bizid] = b

for bb in read_test_file('yelp_test_set_user.json'):
    b = User(bb['user_id'], bb['average_stars'], bb['review_count'], None)
    if b.uid not in UserMap.keys():
        UserMap[b.uid] = b

testReviews = []
for rev in read_test_file('yelp_test_set_review.json'):
    testReviews.append(Review(
        rev['business_id'],
        rev['user_id'],
        rev['stars'],
        rev['text'],
        rev['date'],
        None,
        rev['review_id'],
        ))

def checkUserVotes(uid):
    try:
        return UserMap[uid].VPR()
    except TypeError:
        return None
    except AttributeError:
        return None

d = date(2013, 3, 12)
t = time(0, 0)
measuredDate = datetime.combine(d, t)

collabel = [['id', 'votes']]
resultFile = open('yelpprediction24.csv', 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerows(collabel)

count = 0
err = 0
for rev in testReviews:
    thisBiz = lookUpBiz(rev.bizid)
    thisUser = lookUpUser(rev.uid)
    if thisUser is not None:
        if checkUserVotes(thisUser.uid) is not None:
            dotprod = np.dot(warray, inputfact(rev))
        else:
            dotprod = np.dot(warray2, inputfact2(rev))
    else:
        excount = 0
        for sent in sent_tokenize(rev.text):
            ss = sent.strip()
            if ss.endswith('!'):
                excount += 1
        if len(rev.text) == 0:
            dotprod = np.dot(bizwarray, [
                thisBiz.stars,
                len(rev.text),
                rev.stars,
                abs(rev.date - measuredDate).days,
                0,
                0,
                0,
                0,
                0,
                thisBiz.longitude,
                thisBiz.latitude,
                ])
        else:
            dotprod = np.dot(bizwarray, [
                thisBiz.stars,
                len(rev.text),
                rev.stars,
                abs(rev.date - measuredDate).days,
                excount,
                np.mean([len(sent) for sent in
                        sent_tokenize(rev.text)]),
                len(sent_tokenize(rev.text)),
                len(re.findall('\n\n', rev.text)) + 1,
                len(rev.text.splitlines()[0]),
                thisBiz.longitude,
                thisBiz.latitude,
                ])
    if dotprod < 0:
        dotprod = float(0)
    RESULTS = [[rev.revid, dotprod]]
    wr.writerows(RESULTS)
    count += 1

resultFile.close()
