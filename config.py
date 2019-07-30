N_CLASSES = 2
flms = ['Ontobuilder Term Match', 'AMC Token Path', 'WordNet Jiang Conrath']
# flms =[]
# flms = ['title_title_jac_qgm_3_qgm_3', 'authors_authors_lev_sim']
ts = [0.2]
qs = [0.9, 0.99]
max_seq_length = 30
bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
DATA_COLUMN_A = 'candText'
DATA_COLUMN_B = 'targText'
LABEL_COLUMN = 'exactMatch'
# folds = 5
folds = 5
epochs = 10
batch_size = 32
# dftype = 'ICDM'
dftype = 'Standard'
filename = '../VectorsPO_full.csv'
