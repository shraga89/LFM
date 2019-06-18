import DataHandler as DH
import config as C
import utils as U
import Evaluation as E
import tensorflow as tf
import BertHandler as BH
import DNNs
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time, datetime, os
from tensorflow import keras

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ["CUDA_VISIBLE_DEVICES"] = "01"
config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 28})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)
keras.backend.set_session(sess)

dh = DH.DataHandler('../MixedSmall.csv')
dh.add_thresholded_flms(C.flms, C.ts)
dh.create_answers()
dh.BERT_preprocess(C.max_seq_length)

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.ERROR)

# split = int((len(dh.df)*2)/3)
res = None
eval_res = None
dh.df = dh.df[:10]
kfold = KFold(C.folds, True, 1)
label_list = [0, 1]
i = 1
for train_ix, test_ix in kfold.split(dh.df):
    train = dh.df.ix[train_ix]
    test = dh.df.ix[test_ix]
    eval = E.matchers_evaluation(test, dh.matchers_list, False)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
    print("Starting fold " + str(i) + ' ' + str(st))
    train_labels = U.one_hot(train[C.LABEL_COLUMN], C.N_CLASSES)
    test_labels = U.one_hot(test[C.LABEL_COLUMN], C.N_CLASSES)
    train_multi_labels = dh.answers_bin_missings[train_ix]
    test_multi_labels = dh.answers_bin_missings[test_ix]
    train_mv_labels = dh.answers_mv[train_ix]
    test_mv_labels = dh.answers_mv[test_ix]

    # Instantiate tokenizer
    tokenizer = BH.create_tokenizer_from_hub_module()
    # Feed Pairs:
    train_examples = BH.convert_text_to_examples_pairs(train[C.DATA_COLUMN_A],
                                                       train[C.DATA_COLUMN_B], train[C.LABEL_COLUMN])
    test_examples = BH.convert_text_to_examples_pairs(test[C.DATA_COLUMN_A],
                                                      test[C.DATA_COLUMN_B], test[C.LABEL_COLUMN])

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels_bert) = \
        BH.convert_examples_to_features(tokenizer, train_examples, max_seq_length=C.max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels_bert) = \
        BH.convert_examples_to_features(tokenizer, test_examples, max_seq_length=C.max_seq_length)

    tokenizer = BH.create_tokenizer_from_hub_module()

    # ------------------BERT AS A MATCHER-----------------
    model = DNNs.bert_as_matcher(C.max_seq_length)

    DNNs.initialize_vars(sess)

    test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
                              test, eval, 'BertAsAMatcher', False)
    # ------------------MAJORITY VOTE MODEL---------------
    del model
    model = DNNs.build_model(C.max_seq_length)

    # Instantiate variables
    DNNs.initialize_vars(sess)

    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_mv_labels,
        validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_mv_labels),
        epochs=1,
        batch_size=32
    )
    test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
                              test, eval, 'BertMajority', False)

    # ------------------AGGREGATED MODEL------------------
    del model
    model = DNNs.build_model(C.max_seq_length)

    # Instantiate variables
    DNNs.initialize_vars(sess)
    crowds_agg = CrowdsCategoricalAggregator(model,
                                             [train_input_ids, train_input_masks, train_segment_ids],
                                             dh.answers[train_ix])
    for epoch in range(15):
        print("Epoch:", epoch + 1)

        # E-step
        ground_truth_est = crowds_agg.e_step()
        print("Adjusted ground truth accuracy:",
              1.0 * np.sum(np.argmax(ground_truth_est, axis=1) == train_labels) / len(train_labels))

        # M-step
        model, pi = crowds_agg.m_step()
    test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
                              test, eval, 'BertAggregator', False)

    # ------------------CROWDS----------------------------
    del model
    model = DNNs.build_crowd_model(C.max_seq_length, C.N_CLASSES, dh.N_ANNOT)

    DNNs.initialize_vars(sess)

    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_multi_labels,
        validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_multi_labels),
        epochs=1,
        batch_size=32
    )

    model = DNNs.remove_last_layer(model)
    test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
                              test, eval, 'BertCrowd', False)
    res = pd.concat([res, test], ignore_index=True).drop_duplicates().reset_index(drop=True)
    res = pd.concat([res, eval], ignore_index=True).drop_duplicates().reset_index(drop=True)
    i += 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st
if not os.path.exists(folder):
    os.makedirs(folder)
res.to_csv(folder + '/full_results.csv', index=False)
res.to_csv(folder + '/eval.csv', index=False)
