# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math

def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags = {}
    wordslist = {}
    tagdict = {}
    tagtag = {}
    START = train[0][0]
    END = train[0][len(train[0]) - 1]
    for sent in train:
        for word, tag in sent:
            if tag in tagdict:
                tagdict[tag][word] = tagdict[tag].get(word, 0) + 1
            else:
                tagdict[tag] = {word : 1}
            tags[tag] = tags.get(tag, 0) + 1
            if word in wordslist:
                wordslist[word].append(tag)
            else:
                wordslist[word] = [tag]
        for i in range(len(sent) - 1):
            tag1 = sent[i][1]
            tag2 = sent[i+1][1]
            tagtag[(tag1, tag2)] = tagtag.get((tag1, tag2), 0) + 1
    hapax_ing = {}
    hapax_ly = {}
    hapax_s = {}
    hapax = {}
    ing_count = 0
    ly_count = 0
    s_count = 0
    count = 0
    for word in wordslist:
        if len(wordslist[word]) == 1:
            tag = wordslist[word][0]
            if word.endswith("ing"):
                hapax_ing[tag] = hapax_ing.get(tag, 0) + 1
                ing_count += 1
            elif word.endswith("ly"):
                hapax_ly[tag] = hapax_ly.get(tag, 0) + 1
                ly_count += 1
            elif word.endswith("'s"):
                hapax_s[tag] = hapax_s.get(tag, 0) + 1
                s_count += 1
            else:
                hapax[tag] = hapax.get(tag, 0) + 1
                count += 1
    result = []
    for sent in test:
        list = [START, END]
        dict = {}
        word = sent[1]
        tag_smooth = -400
        for tag in tags:
            if word.endswith("ing"):
                k = 0.00001 * (hapax_ing.get(tag, 0) + 1) / ing_count
            elif word.endswith("ly"):
                k = 0.00001 * (hapax_ly.get(tag, 0) + 1) / ly_count
            elif word.endswith("'s"):
                k = 0.00001 * (hapax_s.get(tag, 0) + 1) / s_count
            else:
                k = 0.00001 * (hapax.get(tag, 0) + 1) / count
            word_smooth = math.log(k / (tags[tag] + k * (len(tagdict[tag]) + 1)))
            if word not in tagdict[tag] and (START[1], tag) not in tagtag:
                dict[(1, tag)] = (word_smooth + tag_smooth, START[1])
            elif word not in tagdict[tag]:
                dict[(1, tag)] = (word_smooth + math.log((tagtag[(START[1], tag)] + k) / (tags[tag] + k * (len(tags) + 1))), START[1])
            elif (START[1], tag) not in tagtag:
                dict[(1, tag)] = (math.log((tagdict[tag][word] + k) / (tags[tag] + k * (len(tagdict[tag]) + 1))) + tag_smooth, START[1])
            else:
                dict[(1, tag)] = (math.log((tagdict[tag][word] + k) / (tags[tag] + k * (len(tagdict[tag]) + 1))) + math.log((tagtag[(START[1], tag)] + k) / (tags[tag] + k * (len(tags) + 1))), START[1])
        for i in range(2, len(sent) - 1):
            for tag in tags:
                word_smooth = math.log(k / (tags[tag] + k * (len(tagdict[tag]) + 1)))
                max = -math.inf
                maxTag = tag
                word = sent[i]
                if word.endswith("ing"):
                    k = 0.00001 * (hapax_ing.get(tag, 0) + 1) / ing_count
                elif word.endswith("ly"):
                    k = 0.00001 * (hapax_ly.get(tag, 0) + 1) / ly_count
                elif word.endswith("'s"):
                    k = 0.00001 * (hapax_s.get(tag, 0) + 1) / s_count
                else:
                    k = 0.00001 * (hapax.get(tag, 0) + 1) / count
                for tag2 in tags:
                    if word not in tagdict[tag] and (tag2, tag) not in tagtag:
                        edge = word_smooth + tag_smooth
                    elif word not in tagdict[tag]:
                        edge = word_smooth + math.log((tagtag[(tag2, tag)] + k) / (tags[tag] + k * (len(tags) + 1)))
                    elif (tag2, tag) not in tagtag:
                        edge = math.log((tagdict[tag][word] + k) / (tags[tag] + k * (len(tagdict[tag]) + 1))) + tag_smooth
                    else:
                        edge = math.log((tagdict[tag][word] + k) / (tags[tag] + k * (len(tagdict[tag]) + 1))) + math.log((tagtag[(tag2, tag)] + k) / (tags[tag] + k * (len(tags) + 1)))
                    if dict[(i-1, tag2)][0] + edge > max:
                        max = dict[(i-1, tag2)][0] + edge
                        maxTag = tag2
                dict[(i, tag)] = (max, maxTag)
        max = -math.inf
        for tag in tags:
            if dict[(len(sent) - 2, tag)][0] > max:
                max = dict[(len(sent) - 2, tag)][0]
                maxTag = tag
        list.insert(1, (sent[len(sent) - 2], maxTag))
        for i in reversed(range(2, len(sent) - 1)):
            maxTag = dict[(i, maxTag)][1]
            list.insert(1, (sent[i-1], maxTag))
        result.append(list)
    return result