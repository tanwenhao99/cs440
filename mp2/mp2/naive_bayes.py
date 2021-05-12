# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    dev_labels = []
    ham = {}
    spam = {}
    ham_count = 0
    spam_count = 0
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            ham_count += len(train_set[i])
            for str in train_set[i]:
                if str in ham:
                    ham[str] += 1
                else:
                    ham[str] = 1
        else:
            spam_count += len(train_set[i])
            for str in train_set[i]:
                if str in spam:
                    spam[str] += 1
                else:
                    spam[str] = 1
    for i in range(len(dev_set)):
        ham_prob = math.log(pos_prior)
        spam_prob = math.log(1 - pos_prior)
        for str in dev_set[i]:
            if str in ham:
                ham_prob += math.log((ham[str] + smoothing_parameter) / (ham_count + smoothing_parameter * len(ham)))
            else:
                ham_prob += math.log(smoothing_parameter / (ham_count + smoothing_parameter * len(ham)))
            if str in spam:
                spam_prob += math.log((spam[str] + smoothing_parameter) / (spam_count + smoothing_parameter * len(spam)))
            else:
                spam_prob += math.log(smoothing_parameter / (spam_count + smoothing_parameter * len(spam)))
        if ham_prob > spam_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels