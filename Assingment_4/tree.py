import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    sh = y.shape
    shape = sh
    H = 0
    p = np.zeros(shape[1])
    for i in range (0, shape[1]):
    	p[i] = np.sum(y[:,i]) / (shape[0] + EPS)
    	H -= p[i] * np.log(p[i] + EPS)       
    return H
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    EPS = 0.0005
    prob = np.sum(y, axis=0)/(len(y)+EPS)
    gini = 1. - np.sum(prob*prob)      
    return gini
    
    
    
#metrics of regression, I don't understand what N we choose
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    Var = 0
    if (y.size == 0):
    	Var = 0
    else:
    	Var = y.var() 
    	
    return Var

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    EPS = 0.0005
    Mad = np.sum(np.abs(y - np.median(y))) / (y.shape[0] + EPS)
    return Mad




def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
   
                
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        
        A = X_subset[:, feature_index] < threshold
        B = A == False
                        
        return (X_subset[A], y_subset[A]), (X_subset[B], y_subset[B])

    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        
        A = X_subset[:, feature_index] < threshold
        B = A == False
        
        return y_subset[A], y_subset[B]


    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        
        num_obj = X_subset.shape[0]
        num_feat = X_subset.shape[1]
        criterion_base = np.inf
        feature_index = 0
        threshold = 0.
        
        if (self.criterion_name == 'gini'):        
        	func = gini
        if (self.criterion_name == 'entropy'):
        	func = entropy
        if (self.criterion_name == 'variance'):
        	func = variance
        if (self.criterion_name == 'mad_median'):
        	func = mad_median
        
        for feature_ind in range (num_feat):
        	for obj in range (num_obj):
        		y_left, y_right = self.make_split_only_y(feature_ind, X_subset[obj, feature_ind], X_subset, y_subset)        		
        		criterion = ((y_left.shape)[0] / num_obj) * func(y_left) + ((y_right.shape)[0] / num_obj) * func(y_right)
        		
        		if criterion < criterion_base:
        			criterion_base = criterion
        			feature_index = feature_ind
        			threshold = X_subset[obj, feature_ind]
        			
        		        
        return feature_index, threshold
  
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        
        if(X_subset.shape[0] == 1 or self.depth == self.max_depth):  
        	self.depth -= 1     	        	
        	new_node.proba = y_subset
                
        else:        	        	
        	(X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        	self.depth += 1
        	
        	new_node.left_child = self.make_tree(X_left, y_left)        	
        	self.depth += 1
        	
        	new_node.right_child = self.make_tree(X_right, y_right)
        	self.depth -= 1  
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)        
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        
        if self.classification:
        	y_predicted = self.predict_proba(X).argmax(axis=1)
        	
        else:
        	num_obj = X.shape[0]
        	y_predicted = np.zeros(num_obj)

        	for obj in range (num_obj): 
        		root = self.root      
        		while ((root.left_child != None) and (root.right_child != None)):
	        		if X[obj][root.feature_index] < root.value:
        				root = root.left_child        			        			        			
	        		else:
        				root = root.right_child
        			
        		y_predicted[obj] = np.sum(root.proba, axis = 0) / root.proba.shape[0]
        
#        	y_predicted = y_predicted_probs[obj].mean()        	
        	        
        return y_predicted
   
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE

        num_obj = X.shape[0]
        y_predicted_probs = np.zeros((num_obj, self.n_classes))
         
        for obj in range (num_obj): 
        	root = self.root      
        	while ((root.left_child != None) and (root.right_child != None)):
	        	if X[obj][root.feature_index] < root.value:
        			root = root.left_child        			        			        			
	        	else:
        			root = root.right_child
        						
        	y_predicted_probs[obj, :] = np.sum(root.proba, axis = 0) / root.proba.shape[0]        	        	
              
        return y_predicted_probs
