import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config1 as Config
import Util
import pickle

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """
        print(embedding_array.shape)

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            """
             
    
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_scores, tf_train_labels))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Predictions
    train_prediction = tf.nn.softmax(model_scores)
    test_prediction = tf.nn.softmax(three_layer_network(tf_test_dataset))

            """
            
            
            self.train_inputs = tf.placeholder(dtype=tf.int32,shape = (Config.batch_size,Config.n_Tokens))
            print(self.train_inputs.get_shape().as_list(),'train_inputs')
            
            self.train_labels = tf.placeholder(dtype = tf.int32, shape = (Config.batch_size,parsing_system.numTransitions()))#num trans
            print(self.train_labels.shape,'train_labels')
            
            self.test_inputs = tf.placeholder(dtype=tf.int32, shape = (Config.n_Tokens))
            
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            print(train_embed.shape,"train_embed shape")
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1]) #Config.n_Tokens*Config.embedding_size
            
            
            weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size_1,Config.embedding_size,Config.n_Tokens],mean=0.0,stddev=0.1))
            weights_input = tf.reshape(weights_input,  [Config.hidden_size_1,-1])
            
            biases_input = tf.Variable(tf.truncated_normal([Config.hidden_size_1,1],mean=0.0,stddev=0.1))
            
            weights_layer_1 = tf.Variable(tf.truncated_normal([Config.hidden_size_1,Config.hidden_size_2],mean=0.0,stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal([Config.hidden_size_2,1],mean=0.0,stddev=0.1))
            
            weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size_2],mean=0.0,stddev=0.1))
            
            self.prediction = self.forward_pass(train_embed, weights_input, biases_input,weights_layer_1,biases_1, weights_output)
#             self.prediction = tf.reshape(self.prediction,[Config.batch_size,-1])
            self.loss = tf.reduce_mean(tf.nn.relu(tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction,labels = self.train_labels)))
          
            # Loss function using L2 Regularization
            self.regularizer = tf.nn.l2_loss(weights_output)+tf.nn.l2_loss(weights_input)+tf.nn.l2_loss(biases_input)+tf.nn.l2_loss(train_embed)+tf.nn.l2_loss(weights_layer_1)+tf.nn.l2_loss(biases_1)
            self.loss = self.loss + (Config.lam/2) * self.regularizer
    
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input,weights_layer_1,biases_1, weights_output)
            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_layer_1, biases_1,weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        print(weights_input.shape,embed.shape, biases_inpu.shape)
#         weights_output = tf.matmul(weights_input,embed) + biases_inpu

        input_layer = tf.matmul(weights_input, tf.transpose(embed))+biases_inpu
        hidden_1 = tf.pow(input_layer,3)
        hidden_2 = tf.matmul(tf.transpose(weights_layer_1), hidden_1)+biases_1
        hidden_2 = tf.pow(hidden_2,3)
        softmax_layer = tf.matmul(weights_output,hidden_2)
        print(hidden_1.shape,'hidden_1')
        print(hidden_2.shape,'hidden_2')
        print(weights_output.shape,'weights_output')
        print(softmax_layer.shape,'softmax')
        return tf.transpose(softmax_layer)
    
    

def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    
    s1 = c.getStack(0)
    s2 = c.getStack(1)
    s3 = c.getStack(2)
    b1 = c.getBuffer(0)
    b2 = c.getBuffer(1)
    b3 = c.getBuffer(2)
    lc1_1 = c.getLeftChild(s1, 1)
    rc2_1 = c.getRightChild(s1, 1)
    lc3_1 = c.getLeftChild(s1, 2)
    rc4_1 = c.getRightChild(s1, 2)
    lc1_2 = c.getLeftChild(s2, 1)
    rc2_2 = c.getRightChild(s2, 1)
    lc3_2 = c.getLeftChild(s2, 2)
    rc4_2 = c.getRightChild(s2,2)
    lc1_lc1 = c.getLeftChild(lc1_1, 1)
    lc2_lc2 = c.getLeftChild(lc1_2, 1)
    rc1_rc1 = c.getRightChild(rc2_1, 1)
    rc2_rc2 = c.getRightChild(rc2_2, 1)
    params = [s1,s2,s3,b1,b2,b3,lc1_1,rc2_1,lc3_1,rc4_1,lc1_2,rc2_2,lc3_2,rc4_2,lc1_lc1,lc2_lc2,rc1_rc1,rc2_rc2]
    features = []
    
    for param in params:
        features.append(getWordID(c.getWord(param)))
    for param in params:
        features.append(getPosID(c.getPOS(param)))
    for param in params[6:]:
        features.append(getLabelID(c.getLabel(param)))
    
    return features




def genTrainExamples(sents, trees):

      
    numTrans = parsing_system.numTransitions()
    features = []
    labels = []
#     sents = [sents[0]]
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                
                c = parsing_system.apply(c, oracle)
    
            
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = '../../word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

