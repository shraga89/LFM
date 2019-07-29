# pip install tensorflow-gpu
# pip install keras
# pip install bert-tensorflow
# pip install tensorflow_hub
# pip install tqdm
import DataHandler as DH
import config as C
import utils as U
import Evaluation as E
import tensorflow as tf
import BertHandler as BH
import DNNs
import DNNsTwo
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time, datetime, os
from tensorflow import keras

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 28})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)
keras.backend.set_session(sess)

dh = DH.DataHandler(C.filename, C.dftype)
dh.add_thresholded_flms(C.flms, C.ts, C.qs)
_, updated_list = E.matchers_evaluation(dh.df, dh.matchers_list, False)
dh.create_answers(updated_list)
dh.BERT_preprocess(C.max_seq_length)

C.epochs = 2

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.ERROR)

# split = int((len(dh.df)*2)/3)
res = None
eval_res = None
# dh.df = dh.df[:10]
kfold = KFold(C.folds, True, 1)
label_list = [0, 1]
i = 1
for train_ix, test_ix in kfold.split(dh.df):
    train = dh.df.ix[train_ix]
    test = dh.df.ix[test_ix]
    eval, _ = E.matchers_evaluation(test, dh.matchers_list, False)
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
    # train_examples = BH.convert_text_to_examples_pairs(train[C.DATA_COLUMN_A],
    #                                                    train[C.DATA_COLUMN_B], train[C.LABEL_COLUMN])
    # test_examples = BH.convert_text_to_examples_pairs(test[C.DATA_COLUMN_A],
    #                                                   test[C.DATA_COLUMN_B], test[C.LABEL_COLUMN])

    train_examplesA = BH.convert_text_to_examples(train[C.DATA_COLUMN_A], train[C.LABEL_COLUMN])
    train_examplesB = BH.convert_text_to_examples(train[C.DATA_COLUMN_B], train[C.LABEL_COLUMN])
    test_examplesA = BH.convert_text_to_examples(test[C.DATA_COLUMN_A], test[C.LABEL_COLUMN])
    test_examplesB = BH.convert_text_to_examples(test[C.DATA_COLUMN_B], test[C.LABEL_COLUMN])

    # Convert to features
    # (train_input_ids, train_input_masks, train_segment_ids, train_labels_bert) = \
    #     BH.convert_examples_to_features(tokenizer, train_examples, max_seq_length=C.max_seq_length)
    # (test_input_ids, test_input_masks, test_segment_ids, test_labels_bert) = \
    #     BH.convert_examples_to_features(tokenizer, test_examples, max_seq_length=C.max_seq_length)

    (train_input_idsA, train_input_masksA, train_segment_idsA, train_labels_bertA) = \
        BH.convert_examples_to_features(tokenizer, train_examplesA, max_seq_length=C.max_seq_length)
    (train_input_idsB, train_input_masksB, train_segment_idsB, train_labels_bertB) = \
        BH.convert_examples_to_features(tokenizer, train_examplesB, max_seq_length=C.max_seq_length)
    (test_input_idsA, test_input_masksA, test_segment_idsA, test_labels_bertA) = \
        BH.convert_examples_to_features(tokenizer, test_examplesA, max_seq_length=C.max_seq_length)
    (test_input_idsB, test_input_masksB, test_segment_idsB, test_labels_bertB) = \
        BH.convert_examples_to_features(tokenizer, test_examplesB, max_seq_length=C.max_seq_length)

    tokenizer = BH.create_tokenizer_from_hub_module()

    # ------------------BERT AS A MATCHER-----------------
    # model = DNNs.bert_as_matcher(C.max_seq_length)
    model = DNNsTwo.bert_as_matcher(C.max_seq_length)
    DNNsTwo.initialize_vars(sess)
    #
    # test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
    #                           test, eval, 'BertAsAMatcher', False)
    test, eval = E.eval_model(model, [test_input_idsA, test_input_masksA, test_segment_idsA] +
                              [test_input_idsB, test_input_masksB, test_segment_idsB]
                              , test_labels,
                              test, eval, 'BertAsAMatcher', True, False)
    # ------------------MAJORITY VOTE MODEL---------------
    # model = DNNs.build_model(C.max_seq_length)
    #
    # # Instantiate variables
    # DNNs.initialize_vars(sess)
    #
    # model.fit(
    #     [train_input_ids, train_input_masks, train_segment_ids],
    #     train_mv_labels,
    #     validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_mv_labels),
    #     epochs=1,
    #     batch_size=32
    # )
    # test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
    # #                           test, eval, 'BertMajority', False)
    # del model
    # model = DNNsTwo.build_model_bert(C.max_seq_length)
    # DNNsTwo.initialize_vars(sess)
    # model.fit(
    #     [train_input_idsA, train_input_masksA, train_segment_idsA] +
    #     [train_input_idsB, train_input_masksB, train_segment_idsB],
    #     train_mv_labels,
    #     validation_data=([test_input_idsA, test_input_masksA, test_segment_idsA] +
    #                      [test_input_idsB, test_input_masksB, test_segment_idsB],
    #                      test_mv_labels),
    #     epochs=C.epochs,
    #     batch_size=C.batch_size
    # )
    # test, eval = E.eval_model(model, [test_input_idsA, test_input_masksA, test_segment_idsA] +
    #                           [test_input_idsB, test_input_masksB, test_segment_idsB]
    #                           , test_labels,
    #                           test, eval, 'BertMajority', False, False)

    del model
    model = DNNsTwo.build_model_bert_lstm(C.max_seq_length)
    DNNsTwo.initialize_vars(sess)
    model.fit(
        [train_input_idsA, train_input_masksA, train_segment_idsA] +
        [train_input_idsB, train_input_masksB, train_segment_idsB],
        train_mv_labels,
        validation_data=([test_input_idsA, test_input_masksA, test_segment_idsA] +
                         [test_input_idsB, test_input_masksB, test_segment_idsB],
                         test_mv_labels),
        epochs=C.epochs,
        batch_size=C.batch_size
    )
    test, eval = E.eval_model(model, [test_input_idsA, test_input_masksA, test_segment_idsA] +
                              [test_input_idsB, test_input_masksB, test_segment_idsB]
                              , test_labels,
                              test, eval, 'BertMajorityLSTM', False, False)
    # ------------------AGGREGATED MODEL------------------
    # del model
    # model = DNNs.build_model(C.max_seq_length)
    #
    # # Instantiate variables
    # DNNs.initialize_vars(sess)
    # crowds_agg = CrowdsCategoricalAggregator(model,
    #                                          [train_input_ids, train_input_masks, train_segment_ids],
    #                                          dh.answers[train_ix])
    # for epoch in range(1):
    #     print("Epoch:", epoch + 1)
    #
    #     # E-step
    #     ground_truth_est = crowds_agg.e_step()
    #     print("Adjusted ground truth accuracy:",
    #           1.0 * np.sum(np.argmax(ground_truth_est, axis=1) == train_labels) / len(train_labels))
    #
    #     # M-step
    #     model, pi = crowds_agg.m_step()
    # test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
    #                           test, eval, 'BertAggregator', False)

    # # ------------------CROWDS----------------------------
    #     model = DNNs.build_crowd_model(C.max_seq_length, C.N_CLASSES, dh.N_ANNOT)
    #
    # DNNs.initialize_vars(sess)
    #
    # model.fit(
    #     [train_input_ids, train_input_masks, train_segment_ids],
    #     train_multi_labels,
    #     validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_multi_labels),
    #     epochs=1,
    #     batch_size=32
    # )
    #
    # model = DNNs.remove_last_layer(model)
    # test, eval = E.eval_model(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels,
    #                           test, eval, 'BertCrowd', False)
    # del model
    # model = DNNsTwo.build_crowd_model_bert(C.max_seq_length, C.N_CLASSES, dh.N_ANNOT)
    # DNNsTwo.initialize_vars(sess)
    # model.fit(
    #     [train_input_idsA, train_input_masksA, train_segment_idsA] +
    #     [train_input_idsB, train_input_masksB, train_segment_idsB],
    #     train_multi_labels,
    #     validation_data=([test_input_idsA, test_input_masksA, test_segment_idsA] +
    #                      [test_input_idsB, test_input_masksB, test_segment_idsB],
    #                      test_multi_labels),
    #     epochs=C.epochs,
    #     batch_size=C.batch_size
    # )
    #
    # model = DNNsTwo.remove_last_layer(model)
    # test, eval = E.eval_model(model, [test_input_idsA, test_input_masksA, test_segment_idsA] +
    #                           [test_input_idsB, test_input_masksB, test_segment_idsB]
    #                           , test_labels,
    #                           test, eval, 'BertCrowd', False, False)
    #
    # del model
    model = DNNsTwo.build_crowd_model_bert_lstm(C.max_seq_length, C.N_CLASSES, dh.N_ANNOT)
    DNNsTwo.initialize_vars(sess)
    model.fit(
        [train_input_idsA, train_input_masksA, train_segment_idsA] +
        [train_input_idsB, train_input_masksB, train_segment_idsB],
        train_multi_labels,
        validation_data=([test_input_idsA, test_input_masksA, test_segment_idsA] +
                         [test_input_idsB, test_input_masksB, test_segment_idsB],
                         test_multi_labels),
        epochs=C.epochs,
        batch_size=C.batch_size
    )

    model = DNNsTwo.remove_last_layer(model)
    test, eval = E.eval_model(model, [test_input_idsA, test_input_masksA, test_segment_idsA] +
                              [test_input_idsB, test_input_masksB, test_segment_idsB]
                              , test_labels,
                              test, eval, 'BertCrowdLSTM', False, False)

    res = pd.concat([res, test], ignore_index=True).drop_duplicates().reset_index(drop=True)
    eval_res = pd.concat([eval_res, eval], ignore_index=True).drop_duplicates().reset_index(drop=True)
    i += 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st
if not os.path.exists(folder):
    os.makedirs(folder)
res.to_csv(folder + '/full_results.csv', index=False)
eval_res.to_csv(folder + '/eval.csv', index=False)
