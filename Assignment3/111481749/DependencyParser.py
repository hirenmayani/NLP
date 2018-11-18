import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Util
import Config
#import ConfigBest as Config


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

    def random_uniform_initializer(self, shape, name, val, trainable=True):
        out = tf.get_variable(shape=list(shape), dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-val, maxval=val, dtype=tf.float32),
                              trainable=trainable, name=name)
        return out

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
#             self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable=False)
 
            layers = Config.layers

            self.train_inputs =  tf.placeholder(dtype=tf.int32, shape=(Config.batch_size,Config.n_Tokens), name='train_inputs')
            self.train_labels = tf.placeholder(dtype=tf.int32, shape=(Config.batch_size,parsing_system.numTransitions()), name='train_labels')
            self.test_inputs =  tf.placeholder(dtype=tf.int32, shape=(Config.n_Tokens), name='test_inputs')


            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.reshape(embed, [Config.batch_size, -1])
            
            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])


            '''
            # weight init by random_uniform_initializer

            weights_input = self.random_uniform_initializer((Config.hidden_size, Config.n_Tokens * Config.embedding_size), "weights_input",0.01, trainable=True)
            biases_input = self.random_uniform_initializer((Config.hidden_size,1), "biases_input", 0.01, trainable=True)
            weights_output = self.random_uniform_initializer((parsing_system.numTransitions(),Config.hidden_size), "weights_output",0.01, trainable=True)
            '''
            
            '''
            # weight init by random normal
            weights_input = tf.Variable(tf.random.normal([Config.hidden_size, Config.n_Tokens * Config.embedding_size],    mean=0.0,stddev=0.005),
                                        name= 'weights_input')
             
            biases_input = tf.Variable( tf.random.normal([Config.hidden_size,1], mean=0.0,stddev=0.005),
                                        name= 'biases_input')
             
            weights_output = tf.Variable(tf.random.normal([parsing_system.numTransitions(),Config.hidden_size], mean=0.0,stddev=0.005),
                                        name= 'weights_output')
            ''' 
                         
            
            l2_loss = None

            # single layer layer configuration            
            if layers ==1 :
            # weight init by trucated normal
                weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens * Config.embedding_size],    mean=0.0,stddev=0.1),
                                            name= 'weights_input')
                
                biases_input = tf.Variable( tf.truncated_normal([Config.hidden_size,1], mean=0.0,stddev=0.005),
                                            name= 'biases_input')
                weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size], mean=0.0,stddev=0.1),
                                            name= 'weights_output')
    
                self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)
                self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)
                l2_loss = tf.math.add_n([
                        tf.nn.l2_loss(embed),
                        tf.nn.l2_loss(weights_input),
                        tf.nn.l2_loss(biases_input),
                        tf.nn.l2_loss(weights_output),                    
                        ],name='adding_l2_losses')
                
                l2_loss = tf.multiply(Config.lam/2.0, l2_loss)  
                          
            # two layer layer configuration            
            elif layers ==2 :
                weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size1, Config.n_Tokens * Config.embedding_size],    mean=0.0,stddev=0.1),
                                            name= 'weights_input')                
                biases_input = tf.Variable( tf.truncated_normal([Config.hidden_size1,1], mean=0.0,stddev=0.005),
                                            name= 'biases_input')
                weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size2, Config.hidden_size1], mean=0.0,stddev=0.1),
                                            name= 'weights_input2')
                biases_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size2,1], mean=0.0,stddev=0.1),
                                            name= 'biases_input2')
                weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size2], mean=0.0,stddev=0.1),
                                            name= 'weights_output')
    
                self.prediction = self.forward_pass2Layers(embed, weights_input, biases_input,weights_input2, biases_input2, weights_output)
                self.test_pred = self.forward_pass2Layers(test_embed, weights_input, biases_input,weights_input2, biases_input2, weights_output)

                l2_loss = tf.math.add_n([
                        tf.nn.l2_loss(embed),
                        tf.nn.l2_loss(weights_input),
                        tf.nn.l2_loss(biases_input),
                        tf.nn.l2_loss(weights_input2),
                        tf.nn.l2_loss(biases_input2),
                        tf.nn.l2_loss(weights_output),                    
                        ],name='adding_l2_losses')
                
                l2_loss = tf.multiply(Config.lam/2.0, l2_loss) 
                           
            # three layer configuration            
            elif layers ==3 :
                weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size1, Config.n_Tokens * Config.embedding_size],    mean=0.0,stddev=0.1),
                                            name= 'weights_input')                
                biases_input = tf.Variable( tf.truncated_normal([Config.hidden_size1,1], mean=0.0,stddev=0.005),
                                            name= 'biases_input')
                weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size2, Config.hidden_size1], mean=0.0,stddev=0.1),
                                            name= 'weights_input2')
                biases_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size2,1], mean=0.0,stddev=0.1),
                                            name= 'biases_input2')
                weights_input3 = tf.Variable(tf.truncated_normal([Config.hidden_size3, Config.hidden_size2],    mean=0.0,stddev=0.005),
                                            name= 'weights_input3')
                biases_input3 = tf.Variable( tf.truncated_normal([Config.hidden_size3,1], mean=0.0,stddev=0.05),
                                            name= 'biases_input3')
                weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size3], mean=0.0,stddev=0.1),
                                            name= 'weights_output2')
    
                self.prediction = self.forward_pass3Layers(embed, weights_input, biases_input,weights_input2, biases_input2,weights_input3, biases_input3, weights_output)
                self.test_pred = self.forward_pass3Layers(test_embed, weights_input, biases_input,weights_input2, biases_input2,weights_input3, biases_input3, weights_output)
                l2_loss = tf.math.add_n([
                        tf.nn.l2_loss(embed),
                        tf.nn.l2_loss(weights_input),
                        tf.nn.l2_loss(biases_input),
                        tf.nn.l2_loss(weights_input2),
                        tf.nn.l2_loss(biases_input2),
                        tf.nn.l2_loss(weights_input3),
                        tf.nn.l2_loss(biases_input3),
                        tf.nn.l2_loss(weights_output),                    
                        ],name='adding_l2_losses')
                
                l2_loss = tf.multiply(Config.lam/2.0, l2_loss)            
            
            # configuration for parallel
            elif layers == 'par' :
                
                weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens1 * Config.embedding_size],    mean=0.0,stddev=0.1),
                                            name= 'weights_input')
                biases_input = tf.Variable( tf.truncated_normal([Config.hidden_size,1], mean=0.0,stddev=0.005),
                                            name= 'biases_input')
                
                weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens2 * Config.embedding_size], mean=0.0,stddev=0.1),
                                            name= 'weights_input2')
                biases_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size,1], mean=0.0,stddev=0.1),
                                            name= 'biases_input2')
                
                weights_input3 = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens3 * Config.embedding_size],    mean=0.0,stddev=0.005),
                                            name= 'weights_input3')
                biases_input3 = tf.Variable( tf.truncated_normal([Config.hidden_size,1], mean=0.0,stddev=0.05),
                                            name= 'biases_input3')
                
                weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size], mean=0.0,stddev=0.1),
                                            name= 'weights_output2')
    
                self.prediction = self.forward_pass_parallel(embed, weights_input, biases_input,weights_input2, biases_input2,weights_input3, biases_input3, weights_output)
                self.test_pred = self.forward_pass_parallel(test_embed, weights_input, biases_input,weights_input2, biases_input2,weights_input3, biases_input3, weights_output)
                
                l2_loss = tf.math.add_n([
                        tf.nn.l2_loss(embed),
                        tf.nn.l2_loss(weights_input),
                        tf.nn.l2_loss(biases_input),
                        tf.nn.l2_loss(weights_input2),
                        tf.nn.l2_loss(biases_input2),
                        tf.nn.l2_loss(weights_input3),
                        tf.nn.l2_loss(biases_input3),
                        tf.nn.l2_loss(weights_output),                    
                        ],name='adding_l2_losses')
                
                l2_loss = tf.multiply(Config.lam/2.0, l2_loss)            
#             self.prediction = tf.Print(self.prediction, [self.prediction], "self.prediction")
#             self.train_labels = tf.Print(self.train_labels, [self.train_labels], "self.train_labels")

            
            # filtering out non-zero results
            correctLabels = tf.nn.relu(self.train_labels)
            # to do this using mask
#             minus1 = tf.constant(-1, dtype=tf.int32)
#             validIndexes = tf.not_equal(self.train_labels, minus1)
#             #validIndexes = validIndexes.eval(session=sess)
#            
#             self.train_labels1 = tf.boolean_mask(self.train_labels, validIndexes)
#             self.prediction1 = tf.boolean_mask(self.prediction,validIndexes)
#             self.prediction1 = tf.Print(self.prediction1, [self.prediction1], "self.prediction1")
#             self.train_labels1 = tf.Print(self.train_labels1, [self.train_labels1], "self.train_labels1")

            

            cross_entropy_loss =  tf.nn.relu(tf.nn.softmax_cross_entropy_with_logits(
                        labels= correctLabels,logits=self.prediction,name='calculate_cross_entropy_loss'))            
#             cross_entropy_loss = tf.Print(cross_entropy_loss, [cross_entropy_loss], "cross_entropy_loss")
             
            
            self.loss = tf.reduce_mean(tf.math.add(cross_entropy_loss, l2_loss))

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)


            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

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
                print ("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print ("\nTesting on dev set at step ", step)
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
                print (result)

        print ("Train Finished.")


    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print ("Starting to predict on test set")
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
        print( "Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass_parallel(self, embed, weights_input, biases_input,weights_input2, biases_input2, 
                            weights_input3, biases_input3, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        two layer forward pass function
        =======================================================
        """
        print('forwadpass inputs')

        print(embed, weights_input, biases_input, weights_input2, biases_input2,weights_input3, biases_input3, weights_output)
        
        embed1,embed2,embed3 = tf.split(embed, [Config.embedding_size*Config.n_Tokens1,Config.embedding_size*Config.n_Tokens2,Config.embedding_size*Config.n_Tokens3], 1)
        print(embed1,embed2,embed3)
        
        layer1 = tf.add(tf.matmul(weights_input, tf.transpose(embed1)),biases_input)
        layer1 = tf.math.pow(layer1,3.0)
        #layer1 = tf.math.tanh(layer1, name = 'tanh_activation1')

        layer2 = tf.add(tf.matmul(weights_input2, tf.transpose(embed2)),biases_input2)
        layer2 = tf.math.pow(layer2,3.0)
        #layer2 = tf.math.tanh(layer2, name = 'tanh_activation2')
        
        layer3 = tf.add(tf.matmul(weights_input3, tf.transpose(embed3)),biases_input2)
        layer3 = tf.math.pow(layer3,3.0)
        #layer3 = tf.math.tanh(layer3 , name = 'tanh_activation3')
        
        print(layer1, layer2,layer3)        
        layer123 = layer1+layer2+layer3 #tf.concat([layer1, layer2,layer3], 0)
        print('--------')
        print('layer123', layer123)
        print('--------')
        print('weights_output', weights_output)
        p = tf.matrix_transpose(tf.matmul(weights_output, layer123))
        print('--------')
        print('p', p)

        return p
    
    
    
    def forward_pass3Layers(self, embed, weights_input, biases_input,weights_input2, biases_input2, 
                            weights_input3, biases_input3, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        two layer forward pass function
        =======================================================
        """
        print('forwadpass inputs')
        print(embed, weights_input, biases_input, weights_input2, biases_input2, weights_output)
        
        layer1 = tf.add(tf.matmul(weights_input, tf.transpose(embed)),biases_input)
        layer1 = tf.math.pow(layer1,3.0)
        #layer1 = tf.math.tanh(layer1, name = 'tanh_activation1')

        layer2 = tf.add(tf.matmul(weights_input2, layer1),biases_input2)
        #layer2 = tf.math.pow(layer2,3.0)
        layer2 = tf.math.tanh(layer2, name = 'tanh_activation2')
        
        layer3 = tf.add(tf.matmul(weights_input3, layer2),biases_input2)
        #layer3 = tf.math.pow(layer3,3.0)
        layer3 = tf.math.tanh(layer3 , name = 'tanh_activation3')
        
        p = tf.matrix_transpose(tf.matmul(weights_output, layer3))
         
        print('p', p)

        return p

    def forward_pass2Layers(self, embed, weights_input, biases_input,weights_input2, biases_input2, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        two layer forward pass function
        =======================================================
        """
        print('forwadpass inputs')
        print(embed, weights_input, biases_input, weights_input2, biases_input2, weights_output)
        
        layer1 = tf.add(tf.matmul(weights_input, tf.transpose(embed)),biases_input)
        layer1 = tf.math.pow(layer1,3.0)

        layer2 = tf.add(tf.matmul(weights_input2, layer1),biases_input2)
        layer2 = tf.math.tanh(layer2)
        

        p = tf.matrix_transpose(tf.matmul(weights_output, layer2))
         
        print('p', p)

        return p
    
    
    def forward_pass(self, embed, weights_input, biases_input, weights_output):
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
        
        print('forwadpass inputs')
        print(embed, weights_input, biases_input, weights_output)
        
        layer1 = tf.add(tf.matmul(weights_input, tf.matrix_transpose(embed)),biases_input)
        layer1 = tf.math.pow(layer1,tf.fill(tf.shape(layer1),3.0))
# tanh activation function
#         layer1 = tf.math.tanh(layer1, name = 'tanh_activation')

# sigmoid activation function
#         layer1 = tf.math.sigmoid(layer1, name = 'sigmoid_activation')

# relu activation function
#         layer1 = tf.nn.relu(layer1, name = 'relu_activation')

        
        print('layer1')
        print(layer1)

        p = tf.matrix_transpose(tf.matmul(weights_output, layer1))
         
        print('p')
        print(p)

        return p

def load_object(filename):
    return pickle.load( open( filename, "rb" ) )

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
    print('dict sizes',len(wordDict),len(posDict), len(labelDict))
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

    lc1_s1 = c.getLeftChild(s1,1)
    rc1_s1 = c.getRightChild(s1,1)
    lc2_s1 = c.getLeftChild(s1,2)
    rc2_s1 = c.getRightChild(s1,2)
    
    lc1_s2 = c.getLeftChild(s2,1)
    rc1_s2 = c.getRightChild(s2,1)
    lc2_s2 = c.getLeftChild(s2,2)
    rc2_s2 = c.getRightChild(s2,2)
    
    lc1_lc1_s1 = c.getLeftChild(lc1_s1, 1)
    rc1_rc1_s1 = c.getRightChild(rc1_s1, 1)

    lc1_lc1_s2 = c.getLeftChild(lc1_s2, 1)
    rc1_rc1_s2 = c.getRightChild(rc1_s2, 1)
    
    indexs = [s1,s2,s3,b1,b2,b3,lc1_s1,rc1_s1,lc2_s1,rc2_s1,lc1_s2,rc1_s2,lc2_s2,
              rc2_s2,lc1_lc1_s1,rc1_rc1_s1,lc1_lc1_s2,rc1_rc1_s2]
    
    set_w = []    
    for i in indexs:
        set_w.append(getWordID(c.getWord(i))) 
    
    set_t = []    
    for i in indexs:
        set_t.append(getPosID(c.getPOS(i))) 

    set_a = []    
    for i in indexs[6:]:
        set_a.append(getLabelID(c.getLabel(i))) 

    features = set_w + set_t + set_a
    return features


def genTrainExamples(sents, trees):
    
    '''
    # to load example preloaded from pkl file uncomment this.
    numTrans = parsing_system.numTransitions()
    try:
        features =load_object("features.pkl")
        labels = load_object("labels.pkl")
        print("loaded from file")
        return features, labels
    except FileNotFoundError:
        pass
    '''
    features = []
    labels = []
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
                    print( i, label)
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
    print ("Found embeddings: ", foundEmbed, "/", len(knownWords))

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

    print(embedding_array.shape)
    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print (parsing_system.rootLabel)

    print ("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print ("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

