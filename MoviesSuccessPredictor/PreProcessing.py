import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PreProccessing():
    regression = None
    def __init__(self, regression=True):
        if regression:
            self.regression = True
            self.moviesFile = pd.read_csv('Movies_training.csv')
        else:
            self.regression = False
            self.moviesFile = pd.read_csv('Movies_training_classification.csv')
        self.moviesFile.drop(['Type'], axis=1, inplace=True)
        self.dataPreproccessing()

    def dataPreproccessing(self):
        # Dropping the cols missing 50% and more
        # nulls = self.check_null(50)
        cols = ['Directors', 'Genres', 'Country', 'Language', 'Runtime', 'Year', 'Age', 'Rotten Tomatoes']
        str_pred= 'rate'
        if self.regression:
           str_pred ='IMDb'

        cols.append(str_pred)
        self.Replace_Missed(cols)
        self.moviesFile['Rotten Tomatoes']= list(map(int, self.moviesFile['Rotten Tomatoes'].str[:-1]))
        self.moviesFile.Year = self.moviesFile.Year.astype("object")
        self.applyOneHotEncoding(['Language', 'Year', 'Directors', 'Genres', 'Country', 'Age'])
        if not self.regression:
            self.applyOneHotEncoding([str_pred])

        self.normalization(['Year', 'Runtime', str_pred , 'Language', 'Directors', 'Genres', 'Country', 'Age', 'Rotten Tomatoes'])
        self.moviesFile.drop(['Title'], axis=1, inplace=True)


    def get_Average(self, idx):
        return self.moviesFile[idx].sum() /len(self.moviesFile[idx])
    def get_Mean(self, idx):
        return self.moviesFile[idx].mean()

    def get_Median(self, idx):
        return  self.moviesFile[idx].median()
    def Replace_Missed(self, cols):
        for col in cols:
            if(self.moviesFile[col].dtype == object):
                most_frq_row=str(self.moviesFile[col].value_counts()[self.moviesFile[col].value_counts() == self.moviesFile[col].value_counts().max()])
                self.moviesFile[col]= self.moviesFile[col].replace(np.nan, most_frq_row.split('   ')[0])
            else:
                self.moviesFile[col]= self.moviesFile[col].replace(np.nan, self.get_Median(col))
        return



    def applyOneHotEncoding(self, columns_bridge):
        for idx in columns_bridge:
            bridge_df = self.OneHotEncoding(idx)
            self.moviesFile[idx] = bridge_df[idx + "_ID"]

    def OneHotEncoding(self, col_bridge):
        bridge_df = pd.DataFrame(self.moviesFile, columns=[col_bridge])
        bridge_df = bridge_df.rename(columns={0: col_bridge + "_ID"})
        PreProccessing.Label_encoder = LabelEncoder()
        bridge_df[col_bridge + "_ID"] = PreProccessing.Label_encoder.fit_transform(bridge_df[col_bridge])
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', categories='auto')
        enc_df = pd.DataFrame(one_hot_encoder.fit_transform(bridge_df[[col_bridge + "_ID"]]).toarray())
        bridge_df = bridge_df.join(enc_df)
        return bridge_df

    def normalization(self, cols):
        for i in cols:
            self.moviesFile[i] = (self.moviesFile[i] - min(self.moviesFile[i])) / (max(self.moviesFile[i]) - min(self.moviesFile[i]))

    def show_data(self, X, y, x_label='X', y_label='Y', title='Data'):
        '''
        Take to columns and draw them
        :param X: the x-axis column
        :param y: the y-axis column
        :return: visual data plot
        '''
        plt.scatter(X, y)
        plt.title = title
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.show()


