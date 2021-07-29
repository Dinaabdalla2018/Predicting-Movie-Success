from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
import time
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier



class Classfication():
    def __init__(self, x_train,y_train, x_test):
        self.X = x_train
        self.y = y_train
        self.X_test = x_test
    def SVC(self, kernel='linear'):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :param kernel: kernel used initially linear
        :return: model used and classified data
        '''
        classification = SVC()
        train_time = 0

        if kernel == 'rbf':
            classification = SVC(kernel='rbf')
            return  self.run_model(classification, str(classification))
        elif kernel == 'poly':
            classification = SVC(kernel='poly')
            return  self.run_model(classification, str(classification))
        else:
            classification = SVC(kernel='linear')
            return  self.run_model(classification, str(classification))


    def decicionTreeClassifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = DecisionTreeClassifier()
        return self.run_model(classification, str(classification))

    def SGDClassifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification= SGDClassifier()
        return self.run_model(classification, str(classification))


    def KNN(self, k):
        '''
        :param k: Determine the number needed to determine which class used
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = KNeighborsClassifier(n_neighbors=k)
        return self.run_model(classification, str(classification))

    def Random_forest_classifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = RandomForestClassifier()
        return self.run_model(classification, str(classification))

    def adaboast_classiefier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''## 4  0.1   150
        classification = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),learning_rate=0.2, algorithm="SAMME",  n_estimators=80)
        return self.run_model(classification, str(classification))

    def Extractor_Classsifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = ExtraTreesClassifier(n_estimators=180)
        return self.run_model(classification, str(classification))

    def Gradient_boost_Classsifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        ## 150 2
        classification = GradientBoostingClassifier(n_estimators=150, max_depth=3)
        return self.run_model(classification, str(classification))
    def gaussian_naive_bayesian_classifier(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = GaussianNB()
        return self.run_model(classification, str(classification))

    def multilayer_preceptron(self, hidden_layers=(100,)):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = MLPClassifier(hidden_layer_sizes=hidden_layers)
        return self.run_model(classification, str(classification))

    def logistic_regression(self, X, y, X_test):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and classified data
        '''
        classification = LogisticRegression()
        return self.run_model(classification, str(classification))


    def run_model(self, model, model_name ):
        train_time = 0
        if (os.path.exists("Saved_Models/Classification/" + model_name.split('(')[0] + ".pkl")):
            model = pickle.load(open("Saved_Models/Classification/"+ model_name.split('(')[0] +".pkl", 'rb'))
        else:
            start_time = time.time()
            # model = RandomForestRegressor()
            model= model.fit(self.X, self.y)
            train_time = round((time.time() - start_time), 4)
            pickle.dump(model, open("Saved_Models/Classification/" + model_name.split('(')[0] + ".pkl", 'wb'))
        start_time = time.time()
        y_predict = model.predict(self.X_test)
        test_time = round((time.time() - start_time), 4)
        return y_predict, train_time, test_time

#iris = datasets.load_iris()
#csw = Classfication()

#w = csw.logistic_regression(iris.data, iris.target, iris.data[:3])
#print(w)
#w = csw.multilayer_preceptron(iris.data, iris.target, iris.data[:3])
#print(w)
