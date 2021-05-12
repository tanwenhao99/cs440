# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math



def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set

    dev_labels = []
    ham_uni = {}
    spam_uni = {}
    ham_bi = {}
    spam_bi = {}
    ham_count = 0
    spam_count = 0
    ham_num = 0
    spam_num = 0
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            ham_count += len(train_set[i])
            ham_num += 1
            if len(train_set[i]) > 0:
                for j in range(len(train_set[i]) - 1):
                    if train_set[i][j] in ham_uni:
                        ham_uni[train_set[i][j]] += 1
                    else:
                        ham_uni[train_set[i][j]] = 1
                    if (train_set[i][j], train_set[i][j + 1]) in ham_bi:
                        ham_bi[(train_set[i][j], train_set[i][j + 1])] += 1
                    else:
                        ham_bi[(train_set[i][j], train_set[i][j + 1])] = 1
                if train_set[i][len(train_set[i]) - 1] in ham_uni:
                    ham_uni[train_set[i][len(train_set[i]) - 1]] += 1
                else:
                    ham_uni[train_set[i][len(train_set[i]) - 1]] = 1
        else:
            spam_count += len(train_set[i])
            spam_num += 1
            if len(train_set[i]) > 0:
                for j in range(len(train_set[i]) - 1):
                    if train_set[i][j] in spam_uni:
                        spam_uni[train_set[i][j]] += 1
                    else:
                        spam_uni[train_set[i][j]] = 1
                    if (train_set[i][j], train_set[i][j + 1]) in spam_bi:
                        spam_bi[(train_set[i][j], train_set[i][j + 1])] += 1
                    else:
                        spam_bi[(train_set[i][j], train_set[i][j + 1])] = 1
                if train_set[i][len(train_set[i]) - 1] in spam_uni:
                    spam_uni[train_set[i][len(train_set[i]) - 1]] += 1
                else:
                    spam_uni[train_set[i][len(train_set[i]) - 1]] = 1
    for i in range(len(dev_set)):
        ham_uni_prob = math.log(pos_prior)
        ham_bi_prob = math.log(pos_prior)
        spam_uni_prob = math.log(1 - pos_prior)
        spam_bi_prob = math.log(1 - pos_prior)
        if len(dev_set[i]) > 0:
            for j in range(len(dev_set[i]) - 1):
                if dev_set[i][j] in ham_uni:
                    ham_uni_prob += math.log((ham_uni[dev_set[i][j]] + unigram_smoothing_parameter) / (ham_count + unigram_smoothing_parameter * len(ham_uni)))
                else:
                    ham_uni_prob += math.log(unigram_smoothing_parameter / (ham_count + unigram_smoothing_parameter * len(ham_uni)))
                if dev_set[i][j] in spam_uni:
                    spam_uni_prob += math.log((spam_uni[dev_set[i][j]] + unigram_smoothing_parameter) / (spam_count + unigram_smoothing_parameter * len(spam_uni)))
                else:
                    spam_uni_prob += math.log(unigram_smoothing_parameter / (spam_count + unigram_smoothing_parameter * len(spam_uni)))
                if (dev_set[i][j], dev_set[i][j + 1]) in ham_bi:
                    ham_bi_prob += math.log((ham_bi[(dev_set[i][j], dev_set[i][j + 1])] + bigram_smoothing_parameter) / (ham_count - ham_num + bigram_smoothing_parameter * len(ham_bi)))
                else:
                    ham_bi_prob += math.log(bigram_smoothing_parameter / (ham_count - ham_num + bigram_smoothing_parameter * len(ham_bi)))
                if (dev_set[i][j], dev_set[i][j + 1]) in spam_bi:
                    spam_bi_prob += math.log((spam_bi[(dev_set[i][j], dev_set[i][j + 1])] + bigram_smoothing_parameter) / (spam_count - spam_num + bigram_smoothing_parameter * len(spam_bi)))
                else:
                    spam_bi_prob += math.log(bigram_smoothing_parameter / (spam_count - spam_num + bigram_smoothing_parameter * len(spam_bi)))
            if dev_set[i][len(dev_set[i]) - 1] in ham_uni:
                ham_uni_prob += math.log((ham_uni[dev_set[i][len(dev_set[i]) - 1]] + unigram_smoothing_parameter) / (ham_count + unigram_smoothing_parameter * len(ham_uni)))
            else:
                ham_uni_prob += math.log(unigram_smoothing_parameter / (ham_count + unigram_smoothing_parameter * len(ham_uni)))
            if dev_set[i][len(dev_set[i]) - 1] in spam_uni:
                spam_uni_prob += math.log((spam_uni[dev_set[i][len(dev_set[i]) - 1]] + unigram_smoothing_parameter) / (spam_count + unigram_smoothing_parameter * len(spam_uni)))
            else:
                spam_uni_prob += math.log(unigram_smoothing_parameter / (spam_count + unigram_smoothing_parameter * len(spam_uni)))
        ham_uni_prob *= 1 - bigram_lambda
        ham_bi_prob *= bigram_lambda
        spam_uni_prob *= 1 - bigram_lambda
        spam_bi_prob *= bigram_lambda
        if ham_uni_prob + ham_bi_prob > spam_uni_prob + spam_bi_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels