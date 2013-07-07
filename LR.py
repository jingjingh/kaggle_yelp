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

TRAINING_DATE = date(2013, 1, 19)
TEST_DATE = date(2013, 3, 12)

class Review(object):
    # mode for training, !mode for test
    def __init__(self, rev, mode):
        self.bizid = rev['business_id']
        self.uid = rev['user_id']
        self.stars = rev['stars']
        self.text = rev['text']
        self.date = datetime.strptime(rev['date'], '%Y-%m-%d')
        self.votes = None
        if 'votes' in rev:
            self.votes = rev['votes']
        self.revid = None
        if 'review_id' in rev:
            self.revid = rev['review_id']
        self.mode = mode

    def get_days(self):
        d = TRAINING_DATE if self.mode else TEST_DATE
        end = datetime.combine(d, time(0, 0))
        return abs(self.date - end).days

class Business(object):
    def __init__(self, bus):
        self.bizid = bus['business_id']
        self.stars = bus['stars']
        self.categories = bus['categories']
        self.longitude = bus['longitude']
        self.latitude = bus['latitude']
        self.reviewCount = bus['review_count'] 

class User(object):
    # mode for training, !mode for test
    def __init__(self, user, mode):
        self.uid = user['user_id']
        self.stars = user['average_stars']
        self.reviewCount = user['review_count']
        self.votes = None
        if 'votes' in user:
            self.votes = user['votes']
        self.mode = mode

    def get_vpr(self):
        if self.mode:
            return self.votes['useful'] / self.reviewCount
        else:
            return None

def find_business(bid):
    if bid in business_map:
        return business_map[bid]
    return None

def find_user(uid):
    if uid in user_map:
        return user_map[uid]
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

business_map = {}
for b in read_training_file('yelp_training_set_business.json'):
    business = Business(b)
    business_map[business.bizid] = business

user_map = {}
for u in read_training_file('yelp_training_set_user.json'):
    user = User(u, True)
    user_map[user.uid] = user

reviews = [Review(r, True) for r in read_training_file('yelp_training_set_review.json')]

vallim = [len(reviews), 0, 0]
allInput = []
for rev in reviews:
    this_business = find_business(rev.bizid)
    excount = 0
    for sent in sent_tokenize(rev.text):
        ss = sent.strip()
        if ss.endswith('!'):
            excount += 1
    if len(rev.text) == 0:
        allInput.append([
            this_business.stars,
            len(rev.text),
            rev.stars,
            rev.get_days(),
            0,
            0,
            0,
            0,
            0,
            this_business.longitude,
            this_business.latitude,
            ])
    else:
        allInput.append([
            this_business.stars,
            len(rev.text),
            rev.stars,
            rev.get_days(),
            excount,
            np.mean([len(sent) for sent in sent_tokenize(rev.text)]),
            len(sent_tokenize(rev.text)),
            len(re.findall('\n\n', rev.text)) + 1,
            len(rev.text.splitlines()[0]),
            this_business.longitude,
            this_business.latitude,
            ])

allUseful = [[rev.votes['useful']] for rev in reviews]
MM = matrix(allInput)
bizLR = (MM.T * MM).I * MM.T
bizw = bizLR * matrix(allUseful)
bizwarray = [bizw[i, 0] for i in range(len(bizw))]


## biz stars, user rc, len(text), stars, days old

def inputfactx(rev, include_vpr):
    this_business = find_business(rev.bizid)
    this_user = find_user(rev.uid)
    result = [ this_business.stars ]
    if include_vpr:
        result += [ this_user.get_vpr() ]
    result += [
        this_user.reviewCount,
        len(rev.text),
        rev.stars,
        rev.get_days() ]
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
    result += [ this_business.longitude, this_business.latitude ]
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
for rev in reviews:
    if rev.votes['useful'] < 6 and rev.get_days() < 1800 and len(rev.text) < 1930:
        tempval += 1
        this_business = find_business(rev.bizid)
        this_user = find_user(rev.uid)
        if this_business is not None and this_user is not None and this_user.reviewCount < 513:
            selectedRevList2.append(rev)
            inputFactor2.append(inputfact2(rev))
            if this_user.get_vpr() < 6:
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

for b in read_test_file('yelp_test_set_business.json'):
    business = Business(b)
    business_map[business.bizid] = business

for u in read_test_file('yelp_test_set_user.json'):
    user = User(u, False)
    if user.uid not in user_map.keys():
        user_map[user.uid] = user


collabel = [['id', 'votes']]
resultFile = open('yelpprediction24.csv', 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerows(collabel)

count = 0
err = 0
test_reviews = [Review(rev, False) for rev in read_test_file('yelp_test_set_review.json')]
for rev in test_reviews:
    this_business = find_business(rev.bizid)
    this_user = find_user(rev.uid)
    if this_user is not None:
        if this_user.get_vpr() is not None:
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
                this_business.stars,
                len(rev.text),
                rev.stars,
                rev.get_days(),
                0,
                0,
                0,
                0,
                0,
                this_business.longitude,
                this_business.latitude,
                ])
        else:
            dotprod = np.dot(bizwarray, [
                this_business.stars,
                len(rev.text),
                rev.stars,
                rev.get_days(),
                excount,
                np.mean([len(sent) for sent in sent_tokenize(rev.text)]),
                len(sent_tokenize(rev.text)),
                len(re.findall('\n\n', rev.text)) + 1,
                len(rev.text.splitlines()[0]),
                this_business.longitude,
                this_business.latitude,
                ])
    if dotprod < 0:
        dotprod = float(0)
    wr.writerows([[rev.revid, dotprod]])
    count += 1

resultFile.close()
