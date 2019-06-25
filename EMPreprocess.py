import numpy as np
import pandas as pd
import py_entitymatching as em
import networkx as nx
# import matplotlib.pyplot as plt
# import os.path
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ds_name = 'dblp_scholar_exp_data'
path = '../ds_with_exact/Structured/' + ds_name + '/exp_data/'


def get_features(c, t):
    feats = {}
    for index, row in features.iterrows():
        a_temp = A[A['id'] == c][row['left_attribute']].values[0]
        b_temp = B[B['id'] == t][row['left_attribute']].values[0]
        feat_name = row['feature_name']
        if a_temp == 'nan' or b_temp == 'nan':
            feats[feat_name] = 0
            continue
        if row['left_attr_tokenizer'] in ('qgm_3', 'dlm_dc0'):
            a_temp = em.feature.tokenizers.tok_qgram(input_string=a_temp, q=3)
            b_temp = em.feature.tokenizers.tok_qgram(input_string=b_temp, q=3)
        feats[feat_name] = sim[row['simfunction']](a_temp, b_temp)
    return feats


print(os.path.isfile(path + 'tableA.csv'))
A = em.read_csv_metadata(path + 'tableA.csv', key='id')
B = em.read_csv_metadata(path + 'tableB.csv', key='id')
train = pd.read_csv(path + 'train.csv', low_memory=False, encoding='ISO-8859-1')
test = pd.read_csv(path + 'test.csv', low_memory=False, encoding='ISO-8859-1')
valid = pd.read_csv(path + 'valid.csv', low_memory=False, encoding='ISO-8859-1')
frames = [train, test, valid]
exact = pd.concat(frames)
exact.columns = ['ltable.id', 'rtable.id', 'gold']
exact.to_csv(path + 'exact.csv', index_label='_id')

print('size of A: ', str(len(A)))
print('size of B: ', str(len(B)))
print('size of exact: ', str(len(exact)))

ob = em.OverlapBlocker()
interest_cols = list(A.columns)
K1 = ob.block_tables(A, B, 'title', 'title',
                     l_output_attrs=interest_cols,
                     r_output_attrs=interest_cols,
                     overlap_size=5)
K1 = ob.block_candset(K1, 'authors', 'authors', overlap_size=3)

sim = em.get_sim_funs_for_matching()
features = em.get_features_for_matching(A.drop('id', axis=1), B.drop('id', axis=1))

K1.to_csv(path + 'K1.csv')
L = em.read_csv_metadata(path + 'K1.csv',
                         key='_id',
                         ltable=A, rtable=B,
                         fk_ltable='ltable_id', fk_rtable='rtable_id')

# L = K1.copy()
# print(L.columns)
L['gold'] = 0
trues = exact[exact['gold'] == 1][['ltable.id', 'rtable.id']]
L['temp'] = L['ltable_id'].astype(str) + L['rtable_id'].astype(str)
trues['temp'] = trues['ltable.id'].astype(str) + trues['rtable.id'].astype(str)
L.loc[L['temp'].isin(trues['temp']), ['gold']] = 1

development_evaluation = em.split_train_test(L, train_proportion=0.5)
development = development_evaluation['train']
evaluation = development_evaluation['test']

train_feature_vectors = em.extract_feature_vecs(development, attrs_after='gold',
                                                feature_table=features)
test_feature_vectors = em.extract_feature_vecs(evaluation, attrs_after='gold',
                                               feature_table=features)

train_feature_vectors = train_feature_vectors.fillna(0.0)
test_feature_vectors = test_feature_vectors.fillna(0.0)

print("tagged pairs:" + str(exact['gold'].value_counts()))

df = pd.DataFrame(columns=['instance', 'candName', 'targName', 'conf', 'realConf'])
epoch = 1
cands = list(exact['ltable.id'])
targs = list(exact['rtable.id'])
block = 1
for c in cands:
    for t in targs:
        e = 0
        if len(exact[(exact['ltable.id'] == c) & (exact['rtable.id'] == t)]['gold'].index) > 0:
            e = float(exact[(exact['ltable.id'] == c) & (exact['rtable.id'] == t)]['gold'])
        feat = get_features(c, t)
        for f in feat:
            full_cand = '.'.join(A[A['id'] == c][interest_cols].drop(['id'], axis=1).astype(str).values.tolist()[0])
            full_targ = '.'.join(B[B['id'] == t][interest_cols].drop(['id'], axis=1).astype(str).values.tolist()[0])
            res_row = np.concatenate((np.array(str(block) + ' ' + str(f)), np.array(str(full_cand)),
                                      np.array(str(full_targ)), np.array(feat[f]), np.array(e)), axis=None)
            df.loc[epoch] = res_row
            epoch += 1

df.to_csv('../' + ds_name + '_em_dataset.csv', index=False)
