from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np


def matchers_evaluation(df, matchers, export = False):
    eval = pd.DataFrame(columns=['Pair', 'Matcher', 'P', 'R', 'F'])
    i = 1
    for pair in df['instance'].unique():
        for m in matchers:
            matcher = df[(df['instance'] == pair)][m]
            exact = df[(df['instance'] == pair)]['exactMatch']
            p, r, f = precision_recall_fscore_support(matcher, exact, average='binary')[:3]
            eval.loc[i] = np.array([pair, m, p, r, f])
            i += 1
    if export:
        eval.sort_values(by='Matcher', ascending=True).to_csv('./matcher_quality.csv', index=False)
    return eval


def eval_model(model, test_data, test_labels, test_ids, eval, name, export = False):
    preds_test = model.predict(test_data)
    preds_test_num = np.argmax(preds_test, axis=1)
    accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)
    no_one_hot_labels = [label[1] for label in test_labels]
    temp = test_ids
    temp['real' + name] = no_one_hot_labels
    temp['pred' + name] = preds_test_num
    temp['pred_non_binary' + name] = [p[1] for p in preds_test]
    i = len(eval)
    for pair in temp['instance'].unique():
        matcher = temp[(temp['instance'] == pair)]['pred' + name]
        exact = temp[(temp['instance'] == pair)]['real' + name]
        p, r, f = precision_recall_fscore_support(matcher, exact, average='binary')[:3]
        eval.loc[i] = np.array([pair, name, p, r, f])
        i += 1
    if export:
        eval.sort_values(by='Matcher', ascending=True).to_csv('./matcher_quality.csv', index=False)
    return pd.DataFrame(temp), eval
