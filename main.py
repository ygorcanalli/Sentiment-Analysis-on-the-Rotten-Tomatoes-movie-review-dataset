#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import operator
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from custom import *
from utils import progress_bar
from torch.autograd import Variable


class Tree:

    def __init__(self, label, parent=None):
        self.label = label
        self.parent = parent
        self.children = []

    def add(self, child):
        self.children.append(child)

#Load English stopwords and covert them to a set
stopwords = [stopword.rstrip('\n') for stopword in open("stopwords.csv")]
stopwords = set(stopwords)

#Covert a sentence/phrase to lowercase, removes punctuation and stopwords
#Returns the empty string if all words are removed from the phrase
def clean_phrase(phrase):
    #Remove punctuation (with a regular expression), convert to lower case, and split the phrase into words
    words = (re.sub("[^a-zA-Z]", " ", phrase)).lower().split()
    #Remove stopwords
    useful_words = [word for word in words if not word in stopwords]
    #Return the phrase back as a string
    return " ".join(useful_words)

#Returns a dictionary with each of the cleaned up phrases in the data and their sentiments
def load_data(file_name):
    #Read the csv file and convert it to a numpy array    
    data = pd.read_csv(file_name, header=0, delimiter="\t", quoting=3)
    data = np.array(data)

    phrases = []
    sentiments = []
    trees = {}
    tree = None

    for _, sid, phrase, sentiment in data:
        phrase = clean_phrase(phrase)
        if not phrase or phrase in phrases:
            continue
        phrases.append(phrase)
        sentiments.append(sentiment)
        try:
            while phrase not in tree.label:
                tree = tree.parent
        except AttributeError:
            trees[sid] = Tree(phrase)
            tree = trees[sid]
            continue
        node = Tree(phrase, tree)
        tree.add(node)
        tree = node

    def walk(tree, depth=0):
        if not tree.children:
            yield tree.label, depth
        else:
            for child in tree.children:
                yield from walk(child, depth + 1)

    depth = {}
    for tree in trees.values():
        depth.update(dict(walk(tree)))

    return {'phrases':phrases, "sentiments":sentiments, "depth": depth}

def create_dictionary(phrases, max_words=None):
    one_word_dictionary = Counter()
    two_word_dictionary = Counter()
    phrase_list_first = {}
    phrase_list_second = {}
    two_word_phrase_list = []

    for phrase in phrases:
        phrase = phrase.split()
        length = len(phrase)
        for i in range(length-1):
            word1 = phrase[i]
            word2 = phrase[i+1]
            words = phrase[i] + " " + phrase[i+1]
            two_word_dictionary[(word1, 1)] += 1
            two_word_dictionary[(word2, 2)] += 1

            phrase_list_first.setdefault(word1, []).append(words)
            phrase_list_second.setdefault(word2, []).append(words)
            two_word_phrase_list.append(words)
            #if phrase_list_second.has_key(word2) and phrase_list_second[word2] != None:
                #print "word2", phrase_list_second[word1]
            #    phrase_list_second[word2] = phrase_list_second[word2].append(words)
            #else:
            #    phrase_list_second[word2] = [words]

        for word in phrase:
            one_word_dictionary[word] += 1


    one_word_dictionary = one_word_dictionary.most_common(2 * max_words)
    two_word_dictionary = two_word_dictionary.most_common(2 * max_words)

    vc1 = pd.DataFrame.from_records(one_word_dictionary)
    vc1 = vc1[2:12]
    s2 = pd.Series(two_word_phrase_list)
    vc2 = s2.value_counts()
    vc2 = vc2[:10]
    ax = vc1.plot(kind="bar")
    vc2.plot(kind="bar", ax=ax, color="red")
    #plt.show()
    
    i = 0
    j = 0
    final_dictionary = set()
    while len(final_dictionary) < max_words:
        #print one_word_dictionary[i]
        #print two_word_dictionary[j]
        if int(one_word_dictionary[i][1])  >= int(two_word_dictionary[j][1]):
            final_dictionary.add(one_word_dictionary[i][0])
            i += 1
        else:
            word_tuple = two_word_dictionary[j][0]
            starting_word = word_tuple[0]
            if word_tuple[1] == 1:
                if starting_word not in phrase_list_first.keys():
                    break
                starts_with = phrase_list_first[starting_word]
                #print starts_with
                for two_word_phrase in starts_with:
                    final_dictionary.add(two_word_phrase)
                    break
                j += 1
            elif word_tuple[1] == 2:
                if starting_word not in phrase_list_second.keys():
                    break
                starts_with = phrase_list_second[starting_word]
                #print starts_with
                for two_word_phrase in starts_with:
                    final_dictionary.add(two_word_phrase)
                    break
                j += 1

    return final_dictionary

#Creates a feature vector as described in the bag of words approach (see paper)
#Setting a max_word will limit the size of the feature vector to that number
#where the dictionary will become the max_words most frequently used words (no setting it includes all words)
def create_feature_vectors(phrases, dictionary, depth, max_words=None):
    vectors = []
    for phrase in phrases:
        vector = []
        local_count = {}
        phrase = phrase.split()

        for word in phrase:
        	try:
	            local_count[word] = 1 / (2**depth[word])
	        except KeyError:
	        	local_count[word] = 1

        for word in dictionary:
            if word in local_count:
                vector.append(local_count[word])
            else:
                vector.append(0)

        vectors.append(vector)

    return np.array(vectors)
    #vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=max_words)
    #feature_vectors = vectorizer.fit_transform(phrases)
    #return feature_vectors.toarray()

#Compares the known sentiment results to the predicted ones and returns the percentage that match
def compute_accuracy(known, predicted):
    size = len(known)
    print(size)
    print(len(predicted))
    if size != len(predicted):
        return -1

    err = 0.0;
    for i in range(size):
        if known[i] != predicted[i]:
            err += 1.0
    return (1.0 - (err/size))

#Trains and test the data using random forests and returns the accuracy 
def run_random_forest(train_vectors, train_results, test_vectors, test_results, n_forests):
    forest = RandomForestClassifier(n_estimators=n_forests).fit(train_vectors, train_results)
    predicted_results = forest.predict(test_vectors)
    return compute_accuracy(test_results, predicted_results)

# Training
def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def run_mlp_net(trainloader, testloader)
    net = MLPNet(3*32*32, 10, [512,1024,1024,800])

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch+200):
        train(trainloader, epoch)
        test(testloader, epoch)

def run_simulation():
    print("Loading training data..."),
    sys.stdout.flush()


    traindata = load_data("train_small.tsv")
    testdata = load_data("validation.tsv")
    print("Done.")
    sys.stdout.flush()

    print("Running random forests...")
    sys.stdout.flush()
    #Computes the models 4 dictionary sizes with 3 forest sizes 
    for i in [10, 1000, 2500]:
        for j in [1, 5, 10]:
            dictionary = create_dictionary(traindata['phrases'], i)
            train_feature_vectors = create_feature_vectors(traindata["phrases"], dictionary, traindata["depth"], max_words=i)
            test_feature_vectors = create_feature_vectors(testdata['phrases'], dictionary, testdata["depth"], max_words=i)
            #print len(train_feature_vectors[1])
            #print len(test_feature_vectors[1])
            traindataset = torch.util.data.TensorDataset(train_feature_vector,
                                                         traindata['sentiments'])
            testdataset = torch.util.data.TensorDataset(test_feature_vectors,
                                                        testdata['sentiments'])
            trainloader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=True, num_workers=2)

            testloader = torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=False, num_workers=2)

           res = run_mlp_net(trainloader, testloader)
            print("(Dictionary words: {} \t Random Forest: {} \t Accuracy: {})".format(i,j,res))
            sys.stdout.flush()

if __name__ == "__main__":
    run_simulation()
