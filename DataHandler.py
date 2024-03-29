import pandas as pd
import numpy as np
from scipy.stats import mode
import config as conf
import utils as U
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataHandler:

    def __init__(self, file, dftype = 'Standard'):
        self.file = file
        self.raw = None
        if dftype == 'Standard':
            self.load_raw_data()
        if dftype == 'ICDM':
            self.load_raw_data_icdm()
        if dftype == 'Entities':
            self.load_raw_data_em()
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

    def load_raw_data_icdm(self):
        mylist = []
        self.raw = None
        for chunk in pd.read_csv(self.file, low_memory=False, chunksize=10 ** 6):
            mylist.append(chunk)
            self.raw = pd.concat(mylist, axis=0)
            self.raw = self.raw.fillna(value='0.0')
        del mylist
        self.raw['conf'] = self.raw['conf'].fillna(0.0)
        self.raw['instance'], self.raw['alg'] = self.raw['instance'].str.replace(",", "+").str.split(pat="+", n=1).str
        self.raw = self.raw.drop(['realConf'], axis=1)
        self.raw['conf'] = self.raw['conf'].astype('float64')

    def load_raw_data_em(self):
        mylist = []
        df = None
        for chunk in pd.read_csv(self.file, low_memory=False, chunksize=10 ** 6):
            mylist.append(chunk)
            df = pd.concat(mylist, axis=0)
            df = df.fillna(value='0.0')
        del mylist
        exact = df[df['instance'] ==
                         '1,title_title_jac_qgm_3_qgm_3'][['instance', 'candName', 'targName', 'realConf']]
        exact['instance'] = '1 exactMatch'
        exact['conf'] = exact['realConf']
        with_exact = pd.concat([df, exact])
        with_exact['instance'] = with_exact['instance'].str.replace(' ', ',')
        self.raw = with_exact
        self.raw['conf'] = self.raw['conf'].fillna(0.0)
        self.raw['instance'], self.raw['alg'] = self.raw['instance'].str.replace(",", "+").str.split(pat="+", n=1).str
        self.raw = self.raw.drop(['realConf'], axis=1)
        self.raw['conf'] = self.raw['conf'].astype('float64')


    def create_dataset(self):
        self.df = pd.DataFrame(pd.pivot_table(data=self.raw.copy(),
                                              index=['instance', 'candName', 'targName'],
                                              columns=['alg'],
                                              values=['conf']
                                              ).reset_index().reset_index())
        self.df.columns = [' '.join(col).strip().replace('conf ', '') for col in self.df.columns.values]
        self.df = self.df.drop(['index'], axis=1)
        self.df = self.df.rename(columns={' exactMatch': 'exactMatch'}, inplace=False)

    def add_thresholded_flms(self, flms, ts, qs):
        for flm in flms:
            for t in ts:
                self.df[flm + '_t=' + str(t)] = np.where(self.df[flm] >= t, 1.0, 0.0)
            for q in qs:
                self.df[flm + '_q=' + str(q)] = np.where(self.df[flm] >= self.df[flm].quantile(q), 1.0, 0.0)
            self.df = self.df.drop([flm], axis=1)
        # print(self.df.columns)
        self.matchers_list = list(self.df.columns.drop(['instance', 'candName', 'targName', 'exactMatch']))
        self.N_ANNOT = len(self.matchers_list)
        for m in [matchers for matchers in self.matchers_list if '+' in matchers]:
            self.df[m] = np.where(self.df[m] > 0.0, 1.0, 0.0)

    def create_answers(self, new_list):
        sub_matcher_list = new_list
        self.N_ANNOT = len(sub_matcher_list)
        self.answers = self.df[sub_matcher_list].values
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
        self.df['candText'] = [''.join([(i+1)*(v + ' ') for i, v in list(enumerate(t.split('.')))]) for t in self.df['candNameText']]
        self.df['candText'] = [' '.join(re.sub(r"([A-Z][a-z])", r" \1", t).replace('_', ' ').split()[0:max_seq_length]).lower() for t in self.df['candText']]
        self.df['targNameText'] = self.df['targName'].tolist()
        self.df['targText'] = [''.join([(i+1)*(v + ' ') for i, v in list(enumerate(t.split('.')))]) for t in self.df['targNameText']]
        self.df['targText'] = [' '.join(re.sub(r"([A-Z][a-z])", r" \1", t).replace('_', ' ').split()[0:max_seq_length]).lower() for t in self.df['targText']]

    def glove_preprocess(self, max_seq_length):
        t = Tokenizer()
        t.fit_on_texts((self.df['candText'] + self.df['targText']).tolist())
        self.df['candEncoded'] = t.texts_to_sequences(self.df['candText'])
        self.df['targEncoded'] = t.texts_to_sequences(self.df['targText'])
        self.df['candEncoded'] = pad_sequences(self.df['candEncoded'], maxlen=max_seq_length, padding='post').tolist()
        self.df['targEncoded'] = pad_sequences(self.df['targEncoded'], maxlen=max_seq_length, padding='post').tolist()
        return t
