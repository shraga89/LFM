N_CLASSES = 2
flms = ['Ontobuilder Term Match', 'AMC Token Path', 'WordNet Jiang Conrath']
ts = [0.2]
qs = [0.5, 0.9, 0.99]
max_seq_length = 256
bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
DATA_COLUMN_A = 'candText'
DATA_COLUMN_B = 'targText'
LABEL_COLUMN = 'exactMatch'
folds = 5
epochs = 5
