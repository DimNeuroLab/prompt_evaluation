import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


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


def get_feature_wise_scores(true_df, pred_df, filler=0):
    accuracy_scores = {}
    f1_scores = {}
    for feature in list(true_df.columns)[1:]:
        y_true = true_df[feature].tolist()
        y_pred = pred_df[feature+'_Y'].to_numpy()
        n_pred = pred_df[feature + '_N'].to_numpy()

        #y_pred[y_pred == -1] = filler
        f_pred=np.stack((n_pred,y_pred))

        f_pred=f_pred.argmax(0)
        accuracy_scores[feature] = accuracy_score(y_true, f_pred)
        f1_scores[feature] = f1_score(y_true, f_pred, average='macro')
    return accuracy_scores, f1_scores


if __name__ == '__main__':
    ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')
    CHAT_GPT = pd.read_csv('output/gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_2_20230923-145901.tsv', sep='\t')

    # block below for feature-wise evaluation
    accuracy_scores, f1_scores = get_feature_wise_scores(ANNOTATIONS, CHAT_GPT)
    print('ACC', accuracy_scores, sep='\n')
    print('F1', f1_scores, sep='\n')

    # uncomment block below for prompt-wise evaluation
    '''
    accuracy_scores, f1_scores = get_prompt_wise_scores(ANNOTATIONS, CHAT_GPT)

    print(accuracy_scores)
    print('ACC', sum(accuracy_scores)/(len(accuracy_scores)))
    print(f1_scores)
    print('F1', sum(f1_scores)/(len(f1_scores)))
    '''