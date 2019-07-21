from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import utils as U
import config as C
from sklearn.metrics.pairwise import cosine_similarity

def matchers_evaluation(df, matchers, export=False):
    eval = pd.DataFrame(columns=['Pair', 'Matcher', 'P', 'R', 'F'])
    i = 1
    for pair in df['instance'].unique():
        for m in matchers:
            matcher = np.where(df[(df['instance'] == pair)][m] > 0.0, 1.0, 0.0)
            exact = np.where(df[(df['instance'] == pair)]['exactMatch'] > 0.0, 1.0, 0.0)
            p, r, f = precision_recall_fscore_support(matcher, exact, average='binary')[:3]
            eval.loc[i] = np.array([pair, m, p, r, f])
            i += 1
    if export:
        eval.sort_values(by='Matcher', ascending=True).to_csv('./matcher_quality.csv', index=False)
    return eval


def eval_model(model, test_data, test_labels, test_ids, eval, name, as_a_matcher = False, export = False):
    preds_test = model.predict(test_data)
    if as_a_matcher:
        pairs = zip(preds_test[0], preds_test[1])
        preds_test = np.array([cosine_similarity(np.mean(a, axis=0).reshape(1, -1),
                                        np.mean(b, axis=0).reshape(1, -1)) for a, b in pairs])
        preds_test_num = np.around(preds_test).reshape(-1, 1)
        non_binary_pred = preds_test.reshape(-1, 1)
    else:
        preds_test_num = np.argmax(preds_test, axis=1)
        non_binary_pred = [p[1] for p in preds_test]
    no_one_hot_labels = np.array([label[1] for label in test_labels])
    temp = test_ids
    temp['real' + name] = no_one_hot_labels
    temp['pred' + name] = preds_test_num
    temp['pred_non_binary' + name] = non_binary_pred
    i = len(eval) + 1
    for pair in temp['instance'].unique():
        matcher = np.where(temp[(temp['instance'] == pair)]['pred_non_binary' + name] > 0.0, 1.0, 0.0)
        exact = temp[(temp['instance'] == pair)]['real' + name]
        p, r, f = precision_recall_fscore_support(matcher, exact, average='binary')[:3]
        eval.loc[i] = np.array([pair, name, p, r, f]).astype('str')
        i += 1
        matcher = temp[(temp['instance'] == pair)]['pred' + name]
        p, r, f = precision_recall_fscore_support(matcher, exact, average='binary')[:3]
        eval.loc[i] = np.array([pair, name, p, r, f]).astype('str')
        i += 1
    if export:
        eval.sort_values(by='Matcher', ascending=True).to_csv('./matcher_quality.csv', index=False)
    return pd.DataFrame(temp.astype(str)), eval
