# %% [code] {"jupyter":{"outputs_hidden":false}}
import numpy as np
import scipy.stats

'''
CategoricalNB: naive bayes model for data which feature variables are categorical
GaussicanNB: naive bayes model for continous numeric variables
'''

class CategoricalNB():
  '''
  Object attributes:
    class_prob: class prior probability, array length n_classes
    feature_value: list with n_features length, each entry is 1D-array of feature's value
    feature_prob: list with n_features lenth, each entry is n_feature_values by n_classes 2D array
                  store likelihood P(X_i | c_j)
  '''
  def fit(self, X, y):
    '''
    Input:
      X: array-like data with shape NxM, where N is numbers of records and M is number of features
      y: array-like target with lenght N
    '''
    X = np.array(X)
    
    self.n_samples = X.shape[0]
    self.n_features = X.shape[1]

    self.class_name = np.unique(y)
    self.n_classes = len(self.class_name)
    self.class_prob = np.zeros(self.n_classes)

    self.feature_value = []
    self.feature_prob = []
    for i in range(self.n_features):
      self.feature_value.append(np.unique(X[:, i]))
      self.feature_prob.append(np.zeros((len(self.feature_value[i]), self.n_classes)))

    #calculate class prior probability and likelihood
    self.cacul_prob(X, y)

  def cacul_prob(self, X, y):
    #calculate class prior probability
    for class_idx in range(self.n_classes):
      p = len(np.where(y == self.class_name[class_idx])[0]) / self.n_samples
      self.class_prob[class_idx] = p 

    #calculate likelihood P(X_i | c_j)
    for i in range(self.n_samples):
      class_index = np.where(self.class_name == y[i])[0][0]
      for j in range(self.n_features):
        feature_value_idx = np.where(self.feature_value[j] == X[i, j])[0][0]
        self.feature_prob[j][feature_value_idx, class_index] += 1  
    for j in range(self.n_features):
      self.feature_prob[j] /= self.class_prob * self.n_samples

  def predict(self, X):
    '''
    Input:
      X: records need to be predicted
         array shape NxM, where N is number of records, M is number of features
    Output:
      pred_class: predicted classes, array lenght N    '''

    n_preds = X.shape[0]
    prob_of_class = np.ones((n_preds, self.n_classes))

    for i in range(self.n_classes):
      for j in range(self.n_features):
        value_index = [np.where(self.feature_value[j] == X[k, j])[0][0] for k in range(n_preds)]
        prob_of_class[:, i] += np.log2([self.feature_prob[j][v, i] for v in value_index])
      prob_of_class[:, i] += np.log2(self.class_prob[i])

    pred = np.argmax(prob_of_class, axis = 1)

    return self.class_name[pred]


class GaussianNB():
  '''
  Object attributes:
    class_prob: array with lenght is number of class, store class prior probability
    means: array with shape n_features by n_classes, store feature means of records in each class
    std: array with shape n_features by n_classes, store feature standard deviation of records in each class
  '''

  def fit(self, X, y):
    '''
    Input:
      X: array-like data with shape NxM, where N is numbers of records and M is number of features
      y: array-like target with lenght N
    '''
    X = np.array(X)

    self.n_samples = X.shape[0]
    self.n_features = X.shape[1]

    self.class_name = np.unique(y)
    self.n_classes = len(self.class_name)
    self.class_prob = np.empty(self.n_classes)

    self.means = np.zeros((self.n_features, self.n_classes))
    self.std = np.zeros((self.n_features, self.n_classes))

    #calculate class prior probability, feature means and standard deviation
    self.cacul_statistic(X, y)

  def cacul_statistic(self, X, y):
    #loop through all class
    for class_idx in range(self.n_classes):
      #get index of records belong to class
      sample_idx = np.where(y == self.class_name[class_idx])[0]

      #calculate prior probability, feature means and standard deviation of this class
      self.means[:, class_idx] = np.mean(X[sample_idx, :], axis = 0)
      self.std[:, class_idx] = np.std(X[sample_idx, :], axis = 0, ddof = 1)
      self.class_prob[class_idx] = len(sample_idx) / self.n_samples

  def predict(self, X):
    '''
    Input: 
      X: records need to be predicted
         array shape NxM, where N is number of records, M is number of features
    Output:
      pred_class: predicted classes, array lenght N

    prob_of_class: array shape N by n_classes, stores calculated probability of each class
    '''

    prob_of_class = np.zeros((X.shape[0], self.n_classes))

    for i in range(self.n_classes):
      for j in range(self.n_features):
        #calculate likelihood P(X_j | c_i)
        likelihood = scipy.stats.norm(self.means[j, i], self.std[j, i]).pdf(X[:, j])

        #tranform to logarithmic function of probability
        prob_of_class[:, i] += np.log2(likelihood)
      prob_of_class[:, i] += np.log2(self.class_prob[i])
    
    pred = np.argmax(prob_of_class, axis = 1)
    return pred