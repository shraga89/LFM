import pandas as pd
import numpy as np
from scipy.stats import mode
import config as conf
import utils as U

class DataHandler:

    def __init__(self, file):
        self.file = file
        self.raw = None
        self.load_raw_data()
        self.df = None
        self.create_dataset()
        self.matchers_list = None
        self.N_ANNOT = None
        self.answers = None
        self.answers_bin_missings = None
        self.answers_mv = None

    def load_raw_data(self):
        mylist = []
        self.raw = None
        for chunk in pd.read_csv(self.file, low_memory=False, chunksize=10 ** 6):
            mylist.append(chunk)
            self.raw = pd.concat(mylist, axis=0)
            self.raw = self.raw.fillna(value='empty')
        del mylist
        self.raw = self.raw.drop(['realConf'], axis=1)

    def create_dataset(self):
        self.df = pd.DataFrame(pd.pivot_table(data=self.raw.copy(),
                                         index=['instance', 'candName', 'targName'],
                                         columns=['alg'],
                                         values=['conf']
                                         ).reset_index().reset_index())
        self.df.columns = [' '.join(col).strip().replace('conf ', '') for col in self.df.columns.values]
        self.df = self.df.drop(['index'], axis=1)

    def add_thresholded_flms(self, flms, ts):
        for flm in flms:
            for t in ts:
                self.df[flm + 't=' + str(t)] = np.where(self.df[flm] >= t, 1.0, 0.0)
            self.df = self.df.drop([flm], axis=1)
        self.matchers_list = list(self.df.columns.drop(['instance', 'candName', 'targName', 'exactMatch']))
        self.N_ANNOT = len(self.matchers_list)
        for m in [matchers for matchers in self.matchers_list if '+' in matchers]:
            self.df[m] = np.where(self.df[m] > 0.0, 1.0, 0.0)

    def create_answers(self):
        self.answers = self.df[self.matchers_list].values
        self.answers_bin_missings = []
        for i in range(len(self.answers)):
            row = []
            for r in range(self.N_ANNOT):
                if self.answers[i, r] == -1:
                    row.append(-1 * np.ones(conf.N_CLASSES))
                else:
                    row.append(U.one_hot(int(self.answers[i, r]), conf.N_CLASSES)[0, :])
            self.answers_bin_missings.append(row)
        self.answers_bin_missings = np.array(self.answers_bin_missings).swapaxes(1, 2)
        self.answers_mv = np.array([U.one_hot(mode(a)[0][0], conf.N_CLASSES) for a in self.answers])
        self.answers_mv = self.answers_mv.reshape(self.answers_mv.shape[0], self.answers_mv.shape[2])

    def BERT_preprocess(self, max_seq_length):
        self.df['candNameText'] = self.df['candName'].tolist()
        self.df['candText'] = [' '.join(t.split('.')[0:max_seq_length]) for t in self.df['candNameText']]
        self.df['targNameText'] = self.df['targName'].tolist()
        self.df['targText'] = [' '.join(t.split('.')[0:max_seq_length]) for t in self.df['targNameText']]

