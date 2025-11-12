import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
from collections import Counter
import time
import random
import math
from sklearn.metrics import accuracy_score, f1_score
import itertools
import csv
from scipy.stats import mode
import math


class Dataset:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.preprocess()

    def preprocess(self):
        self.train_data.drop(columns = ['even'], inplace = True)
        self.test_data.drop(columns = ['even'], inplace = True)


    def preprocess_data(self, image_data, labels):
        image_data = np.array(image_data)/255.0
        labels = np.array(labels) if labels is not None else None
        return image_data, labels

    def classify_wise_data(self, image_data, labels):
        data = {}
        for i in range(len(np.unique(labels))):
            data[i] = []
        for i in range(image_data.shape[0]):
            data[labels[i]].append(image_data[i])
        for k in data.keys():
            data[k] = np.array(data[k])
        return data

    def load_data(self, type='train'):
        if type == 'train':
            image_data, labels = self.train_data.drop(columns = 'label'), self.train_data['label']
            image_data = image_data
        else:
            image_data, labels = self.test_data.drop(columns='label'), self.test_data['label']
            labels = pd.Series(labels)
            valid_mask = ~labels.isna()

            print("Dropping", int((~valid_mask).sum()), "rows with missing labels")

            image_data_clean = image_data[valid_mask.values]
            labels_clean = labels[valid_mask].astype(int).to_numpy()
            image_data = image_data_clean

        image_data, labels = self.preprocess_data(image_data, labels)
        if labels is None:
            return image_data
        m = image_data.shape[0]
        image_data = image_data.reshape(m,-1)
        data = self.classify_wise_data(image_data, labels)
        return image_data, labels, data

class SVM:
    def __init__(self, total_class, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
        self.total_class = total_class

    def hinge_loss(self, W, b, X, Y):
        loss = 0.0
        loss += .5 * np.dot(W, W.T)
        m = X.shape[0]
        for i in range(m):
            ti = Y[i] * (np.dot(W,X[i].T) + b)
            loss += self.C * max(0,(1 - ti))
        return loss[0][0]
    def fit(self, X, Y, batch_size=50, learning_rate=0.001, max_iter=500):
        print(X.shape, Y.shape)
        num_features = X.shape[1]
        num_samples = X.shape[0]
        n = learning_rate
        c = self.C
        W = np.zeros((1, num_features))
        bias = 0
        losses = []
        for i in tqdm(range(max_iter)):
            l = self.hinge_loss(W, bias, X, Y)
            losses.append(l)
            ids = np.arange(num_samples)
            np.random.shuffle(ids)
            for batch_start in range(0, num_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_start, batch_start + batch_size):
                    if j < num_samples:
                        i = ids[j]
                        ti =  Y[i] * (np.dot(W, X[i].T) + bias)
                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c * Y[i] * X[i]
                            gradb += c * Y[i]
                W = W - n * W + n * gradw
                bias = bias + n * gradb
        self.W = W
        self.b = bias
        return W, bias, losses

    def get_data_pair(self, data1,data2):
      len1, len2 = data1.shape[0], data2.shape[0]
      samples = len1 + len2
      features = data1.shape[1]
      data_pair = np.zeros((samples,features))
      data_labels = np.zeros((samples,))
      data_pair[:len1,:] = data1
      data_pair[len1:,:] = data2
      data_labels[:len1] = -1
      data_labels[len1:] = +1
      return data_pair, data_labels

    def train(self, data, batch_size=50, learning_rate=0.00001, max_iter=500):
        self.svm_classifiers = {}
        for i in range(self.total_class):
            self.svm_classifiers[i] = {}
            for j in range(i + 1, self.total_class):
                xpair, ypair = self.get_data_pair(data[i], data[j])
                wts, b, loss = self.fit(xpair, ypair, batch_size = batch_size, learning_rate=learning_rate, max_iter=max_iter)
                self.svm_classifiers[i][j] = (wts, b)

        return loss

    def binary_predict(self, x, w, b):
        z  = np.dot(x, w.T) + b
        return 1 if z >= 0 else -1

    def predict(self, x):
        count = np.zeros((self.total_class,))
        for i in range(self.total_class):
            for j in range(i + 1, self.total_class):
                w, b = self.svm_classifiers[i][j]
                z = self.binary_predict(x, w, b)
                if z == 1:
                    count[j] += 1
                else:
                    count[i] += 1
        final_prediction = np.argmax(count)
        return final_prediction

    def accuracy(self, x, y):
        count = 0
        predictions = []
        for i in range(x.shape[0]):
            prediction = self.predict(x[i])
            predictions.append(prediction)
            if(prediction == y[i]):
                count += 1
        return count / x.shape[0], predictions

def plot_confusion_matrix(label_list,pred_list):
    cm_percentage=confusion_matrix(label_list,pred_list, normalize='true')
    cm_number=confusion_matrix(label_list,pred_list)
    df_cm = pd.DataFrame(cm_percentage*100, range(10), range(10))
    plt.figure(figsize=(16,8))
    sn.set(font_scale=1.4)
    ax = sn.heatmap(df_cm, annot=True,cmap='coolwarm', annot_kws={"size": 16}, fmt='.2f')
    for t in ax.texts: t.set_text(t.get_text() + " %")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    cm_number = cm_number.astype(int)
    df_cm = pd.DataFrame(cm_number, range(10), range(10))
    plt.figure(figsize=(12,8))
    sn.set(font_scale=1.4)
    ax = sn.heatmap(df_cm, annot=True, cmap='coolwarm', annot_kws={"size": 16}, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


def evaluate(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average='macro')
    )

def predict(model, data):
    predictions = []
    for image in tqdm(data):
        pred = model.predict(image)
        predictions.append(pred)
    return predictions

class simple_knn():

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            k_closest_y = labels[:k]
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)

    def compute_distances(self,X):
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]

      dot_pro = np.dot(X, self.X_train.T)
      sum_square_test = np.square(X).sum(axis = 1)
      sum_square_train = np.square(self.X_train).sum(axis = 1)
      dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

      return(dists)

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # index of feature to split on
        self.threshold = threshold          # threshold value to split
        self.left = left                    # left subtree
        self.right = right                  # right subtree
        self.value = value                  # class label for leaf nodes

    def is_leaf_node(self):
        # returns true if this node hold a value
        return self.value is not None

class XGBoostTree:
    def __init__(self, max_depth=3,lambda_reg=1.0, gamma=0.0, feature_indices=None, min_samples_split=2):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.feature_indices = feature_indices
        self.root = None
        self.min_samples_split = min_samples_split


    def fit(self, X, g, h):
        X = np.asarray(X)
        self.n_features = X.shape[1]
        if self.feature_indices is None:
            self.feature_indices = list(range(self.n_features))
        self.root = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        n_samples = X.shape[0]
        if n_samples == 0:
            return None

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._leaf_weight(g, h)
            return DecisionTreeNode(value=leaf_value)

        best_gain = -np.inf
        best_idx, best_thresh = None, None
        best_masks = None
        G = np.sum(g)
        H = np.sum(h)

        for feat in self.feature_indices:
            col = X[:, feat]
            thresholds = np.unique(col)
            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                G_L = np.sum(g[left_mask])
                H_L = np.sum(h[left_mask])
                G_R = G - G_L
                H_R = H - H_L
                gain = 0.5 * (
                    (G_L**2) / (H_L + self.lambda_reg) +
                    (G_R**2) / (H_R + self.lambda_reg) -
                    (G**2) / (H + self.lambda_reg)
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_idx = feat
                    best_thresh = thresh
                    best_masks = (left_mask, right_mask)

        if best_gain == -np.inf or best_gain <= 0:
            leaf_value = self._leaf_weight(g, h)
            return DecisionTreeNode(value=leaf_value)

        left_mask, right_mask = best_masks
        left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return DecisionTreeNode(feature_index=best_idx, threshold=best_thresh, left=left, right=right)

    def _leaf_weight(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        if H + self.lambda_reg == 0:
            return 0.0
        return - G / (H + self.lambda_reg)

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([self._predict_row(row, self.root) for row in X])
        return preds

    def _predict_row(self, row, node):
        if node.is_leaf_node():
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

def softmax(F):
  # F: (n_samples, n_classes)
  F_max = np.max(F, axis=1, keepdims=True)
  expF = np.exp(F - F_max)
  denom = np.sum(expF, axis=1, keepdims=True)
  return expF / denom

def one_hot(y, num_class):
  n = y.shape[0]
  Y = np.zeros((n, num_class), dtype=float)
  Y[np.arange(n), y] = 1.0
  return Y


def grad_hess_multiclass(Y_onehot, F):
    # Y_onehot: (n, K), F: (n, K)
    P = softmax(F)
    G = P - Y_onehot
    H = P * (1 - P)
    return G, H, P

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_reg=1.0, gamma=0.0,
                 max_features=25, subsample=1.0, random_state=None,num_class=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.max_features = max_features
        self.subsample = subsample
        self.trees = []
        self.base_score = None
        self.random_state = random_state
        self.num_class = num_class
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        if self.num_class is None:
            self.num_class = int(np.max(y) + 1)

        K = self.num_class
        F = np.zeros((n_samples, K), dtype=float)
        Y_onehot = one_hot(y, K)

        self.trees = []
        for t in range(self.n_estimators):
            G, H, P = grad_hess_multiclass(Y_onehot, F)

            trees_this_round = []
            for k in range(K):
                gk = G[:, k]
                hk = H[:, k]

                feat_idxs = None
                if self.max_features is not None and self.max_features < X.shape[1]:
                    feat_idxs = random.sample(range(X.shape[1]), int(self.max_features))

                if self.subsample is not None and self.subsample < 1.0:
                    row_mask = np.random.rand(n_samples) <= self.subsample
                    if np.sum(row_mask) == 0:
                        row_mask = np.ones(n_samples, dtype=bool)
                    X_sub = X[row_mask]
                    gk_sub = gk[row_mask]
                    hk_sub = hk[row_mask]
                else:
                    X_sub = X
                    gk_sub = gk
                    hk_sub = hk

                tree = XGBoostTree(max_depth=self.max_depth, lambda_reg=self.lambda_reg, gamma=self.gamma,
                                   feature_indices=feat_idxs)
                tree.fit(X_sub, gk_sub, hk_sub)
                trees_this_round.append(tree)
                pred_update = tree.predict(X)
                F[:, k] += self.learning_rate * pred_update

            self.trees.append(trees_this_round)

        self.raw_prediction_ = F


    def predict_proba(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        K = self.num_class
        F = np.zeros((n, K), dtype=float)
        for trees_round in self.trees:
            for k, tree in enumerate(trees_round):
                F[:, k] += self.learning_rate * tree.predict(X)

        P = softmax(F)
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

