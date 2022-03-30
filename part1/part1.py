import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
        
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        sum = 0
        e = 0
        list1 = np.unique(Y)

        counts = np.zeros(len(list1))

        for a in range(0, len(list1)):
            for b in range(0, len(Y)):
                if Y[b] == list1[a]:
                    counts[a] += 1

        for a in range(0, len(counts)):
            sum += counts[a]

        for a in range(0, len(counts)):
                e += -((counts[a] / sum) * math.log2(counts[a] / sum))

        #########################################
        return e



    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce = 0
        x = np.unique(X)
        X_new = []
        temp = []

        for a in range(0, len(x)):
            for b in range(0, len(X)):
                if X[b] == x[a]:
                    temp.append(Y[b])
            X_new.append(temp)
            temp = []

        for a in range(0, len(x)):
            ce += (len(X_new[a])/len(X))*Tree.entropy(X_new[a])

        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        g = Tree.entropy(Y) - Tree.conditional_entropy(Y,X)

        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        i = 0
        temp = []
        info_gain = []
        for a in range(0, len(X)):
            k = Tree.information_gain(Y,X[a])
            info_gain.append(k)

        max_value = max(info_gain)
        i = info_gain.index(max_value)
        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        values = np.unique(X[i])
        x_new = []
        y_new = []
        C = {}

        for a in range(0, len(values)):
            temp = 0
            for k in range(0, np.shape(X)[0]):
                for b in range(0, len(Y)):
                    if X[i][b] == values[a]:
                        x_new = np.append(x_new,X[k][b])
                        if k == 0:
                            temp+=1
                            y_new = np.append(y_new,Y[b])

            x_new = np.reshape(x_new,(np.shape(X)[0],temp))
            y_new = np.reshape(y_new,(temp,))
            node = Node(x_new,y_new)
            C[values[a]] = node
            x_new = []
            y_new = []

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''

        #########################################
        ## INSERT YOUR CODE HERE

        s = np.all(Y == Y[0])
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar.
        '''

        #########################################
        ## INSERT YOUR CODE HERE

        count  = 0
        for a in range(X.shape[0]):
            for b in range(X.shape[1]):
                if np.all(X[a][b] == X[a][0]):
                    count = count+1

        if count == (X.shape[0] * X.shape[1]):
            s = True
        else:
            s = False

        #########################################
        return s
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        counts = Counter(Y)
        k = counts.most_common(1)
        y = k[0][0]
        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        if t.isleaf == False:
            if Tree.stop1(t.Y) == 1:
                #print(Tree.stop1(t.Y))
                t.isleaf = True
                t.p = Tree.most_common(t.Y)

            elif Tree.stop2(t.X):
                #print(Tree.stop2(t.X))
                t.isleaf = True
                t.p = Tree.most_common(t.Y)

            else:
                t.i = Tree.best_attribute(t.X, t.Y)
                t.C = Tree.split(t.X, t.Y, t.i)
                t.p = Tree.most_common(t.Y)

                for key in t.C.keys():
                    K = t.C[key]
                    #K.p = Tree.most_common(K.Y)
                    Tree.build_tree(K)

        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        t = Node(X, Y)
        Tree.build_tree(t)
        #########################################
        return t

    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vector of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        y = t.p
        if t.isleaf == False:
            for key in t.C.keys():
                for a in x:
                    if key == a:
                        if t.C[key].isleaf == True:
                            y = t.C[key].p
                        else:
                            y = Tree.inference(t.C[key], x)

        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        x_new = []
        Y = []
        for a in range(0, np.shape(X)[1]):
            for b in range(0, np.shape(X)[0]):
                x_new.append(X[b][a])
            y = Tree.inference(t,x_new)
            Y = np.append(Y,y)
            x_new = []
        #########################################
        return Y

    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        x_temp = np.genfromtxt( filename , skip_header=1, delimiter = ',', dtype=str)
        x = x_temp[0:,1:]
        X = np.transpose(x)
        #print(X)
        Y = np.loadtxt( filename , skiprows=1, delimiter = ',', usecols=[0], dtype='str')
        #print(Y)
        #########################################
        return X,Y



