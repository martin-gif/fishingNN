import tensorflow as tf
import keras
from sklearn import tree


def get_Decision_Tree():
    clf = tree.DecisionTreeClassifier()
    return clf
