import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from evaluate_chatgpt_prob import evaluate_prob


def get_prompt_wise_scores(true_df, pred_df, filler=0):
    accuracy_scores = []
    f1_scores = []

    for idx, row in true_df.iterrows():
        y_true = row.values[1:]
        y_pred = pred_df.iloc[idx].values[1:]
        y_pred[y_pred == -1] = filler

        y_true = list(y_true)
        y_pred = list(y_pred)

        accuracy_scores.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))

    return accuracy_scores, f1_scores


def get_feature_wise_scores(true_df, pred_df, filename, filler=0):
    accuracy_scores = {}
    f1_scores = {}

    accuracy_scores['run'] = f
    f1_scores['run'] = f

    for feature in list(true_df.columns)[1:]:
        y_true = true_df[feature].tolist()
        y_pred = pred_df[feature + '_Y'].to_numpy()
        n_pred = pred_df[feature + '_N'].to_numpy()

        f_pred = np.stack((n_pred, y_pred))
        f_pred = f_pred.argmax(0)

        accuracy_scores[feature] = accuracy_score(y_true, f_pred)
        f1_scores[feature] = f1_score(y_true, f_pred, average='macro')

    return accuracy_scores, f1_scores


if __name__ == '__main__':
    ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')
    files = glob.glob("output/*log_2shots*.tsv")

    evaluate_prob(ANNOTATIONS, files)
