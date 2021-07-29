import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import seaborn as sns
from Regression_Models import Regression
from Classfication_Models import Classfication
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from PreProcessing import *
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


class Logic:
    def __init__(self,  data_set, pre, regression=True):
        self.data_set = data_set
        self.pre = pre
        if regression:
            self.data = self.get_top_features('IMDb')
            self.x_train, self.x_test, self.y_train, self.y_test = self.split(np.asarray(self.data))
            self.Trainig_phase_regression()
        else:
            self.data = self.get_top_features('rate')
            self.x_train, self.x_test, self.y_train, self.y_test = self.split(np.asarray(self.data))
            self.Training_phase_classification()

    def split(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    def get_top_features(self, column):
        corr =self.data_set.corr()
        top_feature = corr.index[abs(corr[column] > 0)]
        top_corr = self.data_set[top_feature].corr()
        sns.heatmap(top_corr, annot=True)
        plt.show()
        return self.data_set


    def Trainig_phase_regression(self):
        ######################################################################################
        print('######################       Regression         ###################### ')
        print('######################  Random forest Reression ###################### ')
        reg= Regression(self.x_train, self.y_train, self.x_test)
        self.print_model_output(reg.Random_forest_regressor(), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        print('######################  Decision tree Reression ###################### ')
        self.print_model_output( reg.Decision_tree_regressor(), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        print('###################### Polynomaial Reression ######################')
        self.print_model_output(reg.Polynomial(), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        print('######################  Linear Reression ###################### ')
        self.print_model_output(reg.Linear(), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        print('######################  Bayesian_Ridge Reression ###################### ')
        self.print_model_output(reg.Bayesian_Ridge(), True)
        print('-------------------------------------------------------------------')

        #######################################################################################
        print('######################  Lasso Reression ###################### ')
        self.print_model_output(reg.Lasso_Regression(), True)
        print('-------------------------------------------------------------------')

        #######################################################################################
        print('######################  Ridge Reression ###################### ')
        self.print_model_output(reg.Ridge_Regression(), True)
        print('-------------------------------------------------------------------')

        ########################################################################################
        print('###################### RBF SVR Reression ###################### ')
        self.print_model_output(reg.SVR(kernel='rbf'), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        print('###################### Polynomial SVR Reression ###################### ')
        self.print_model_output(reg.SVR( kernel='poly'), True)
        print('-------------------------------------------------------------------')

        ######################################################################################
        #print('###################### Linear SVR Reression ###################### ')
        #model, y_predict = Regression().SVR(self.x_train, self.y_train, self.x_test, kernel='linear')
        #self.Testing_phase(model, y_predict)
        #print('-------------------------------------------------------------------')

    def Testing_phase(self, y_test_predicted, regression=True):
        true_movie_rating = 0.0
        predicted_movies_rating = 0.0
        if regression:
            print('Mean Square Error', metrics.mean_squared_error(self.y_test, y_test_predicted))
            true_movie_rating = np.asarray(self.y_test)[0]
            predicted_movies_rating = y_test_predicted[0]
            r2_Score= r2_score(self.y_test, y_test_predicted)
            print('R2_Score: ', r2_Score)
        else:
            print('Accuracy : ', accuracy_score(self.y_test, y_test_predicted))
            true_movie_rating = self.pre.Label_encoder.inverse_transform([np.asarray(self.y_test)[0]])[0]
            predicted_movies_rating = self.pre.Label_encoder.inverse_transform([y_test_predicted[0]])[0]


        print('True value for the first movie in the test set is : ' + str(true_movie_rating))
        print('Predicted value for the first movie in the test set is : ' + str(predicted_movies_rating))


    def Training_phase_classification(self):
        self.y_train = self.label_encoder(self.y_train)
        self.y_test = self.label_encoder(self.y_test)
        classification =Classfication(self.x_train, self.y_train, self.x_test)
        print('###################### Classification ######################## ')
        print('-------------------------------------------------------------- ')
        print('###################### GaussianNB ###################### ')
        self.print_model_output(classification.gaussian_naive_bayesian_classifier())
        print('-------------------------------------------------------------------')

        print('###################### KNN ###################### ')
        self.print_model_output(classification.KNN(7))
        print('-------------------------------------------------------------------')

        #print('###################### logistic ###################### ')
        #model, y_predict = Classfication().logistic_regression(self.x_train, self.y_train, self.x_test)
        #self.Testing_phase(model, y_predict, regression=False)
        #print('-------------------------------------------------------------------')

        print('###################### SGD ###################### ')
        self.print_model_output( classification.SGDClassifier())
        print('-------------------------------------------------------------------')

        print('###################### adaboast ###################### ')
        self.print_model_output(classification.adaboast_classiefier())
        print('-------------------------------------------------------------------')

        print('###################### random forest ###################### ')
        self.print_model_output(classification.Random_forest_classifier())
        print('-------------------------------------------------------------------')

        print('###################### decision ###################### ')
        self.print_model_output(classification.decicionTreeClassifier())
        print('-------------------------------------------------------------------')

        print('###################### multi preceptron ###################### ')
        self.print_model_output(classification.multilayer_preceptron())
        print('-------------------------------------------------------------------')

        print('###################### svc rbf ###################### ')
        self.print_model_output(classification.SVC(kernel='rbf'))
        print('-------------------------------------------------------------------')

        print('###################### svc poly ###################### ')
        self.print_model_output(classification.SVC(kernel='poly'))
        print('-------------------------------------------------------------------')

        print('###################### Gradient_boost ###################### ')
        self.print_model_output(classification.Gradient_boost_Classsifier())
        print('-------------------------------------------------------------------')

        print('###################### Extractor_Classsifier ###################### ')
        self.print_model_output(classification.Extractor_Classsifier())
        print('-------------------------------------------------------------------')

        #print('###################### svc linear ###################### ')
        #model, y_predict = Classfication().SVC(self.x_train, self.y_train, self.x_test, kernel='linear')
        #self.Testing_phase(model, y_predict, regression=False)
        #print('-------------------------------------------------------------------')

    def label_encoder(self, trainingScores):
        lab_enc = LabelEncoder()
        encoded = lab_enc.fit_transform(trainingScores)
        return encoded


    def print_model_output(self, model, reg=False):
        y_predict, train_time, test_time = model
        print("Training time: %s" % train_time)
        print("Testing time: %s" % test_time)
        self.Testing_phase(y_predict, regression=reg)
