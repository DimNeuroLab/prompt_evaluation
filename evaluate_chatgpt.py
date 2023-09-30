import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import glob

def get_prompt_wise_scores(true_df, pred_df, f, filler=0):
    accuracy_scores['run'] = f
    f1_scores['run'] = f
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


def get_feature_wise_scores(true_df, pred_df,f, filler=0):
    accuracy_scores = {'run': f}
    f1_scores= {'run': f}

    # f1_scores = {}
    for feature in list(true_df.columns)[1:]:
        y_true = true_df[feature].tolist()
        y_pred = pred_df[feature].to_numpy()
        y_pred[y_pred == -1] = filler
        accuracy_scores[feature] = accuracy_score(y_true, y_pred)
        f1_scores[feature] = f1_score(y_true, y_pred, average='macro')
    return accuracy_scores, f1_scores


if __name__ == '__main__':
    ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')
    pred_selection_string = "pred_ense_majority_*"
    files =  glob.glob("output/" + pred_selection_string + ".tsv")
    #CHAT_GPT = pd.read_csv('output/chatgpt_evaluation_1shot.tsv', sep='\t')

    features_results_accuracy = []
    features_results_f1 = []
    prompt_results_accuracy = []
    prompt_results_f1 = []

    for f in files:
        CHAT_GPT = pd.read_csv(f, sep='\t')
        print('f')
        print(f)
        print(CHAT_GPT)

        # block below for feature-wise evaluation
        accuracy_scores, f1_scores = get_feature_wise_scores(ANNOTATIONS, CHAT_GPT,f)
        print('ACC', accuracy_scores, sep='\n')
        print('F1', f1_scores, sep='\n')
        features_results_f1.append(f1_scores)
        features_results_accuracy.append(accuracy_scores)

    result_data = pd.DataFrame(features_results_f1)
    result_data.to_csv('output\eval_det_runs_f1'+pred_selection_string.replace('*',"OO")+'.tsv',
                       sep='\t', index=False)
    result_data = pd.DataFrame(features_results_accuracy)
    result_data.to_csv('output\eval_det_runs_acc_'+pred_selection_string.replace('*',"OO")+'.tsv',
                       sep='\t', index=False)

    # block below for feature-wise evaluation
#    accuracy_scores, f1_scores = get_feature_wise_scores(ANNOTATIONS, CHAT_GPT)
 #   print('ACC', accuracy_scores, sep='\n')
 #   print('F1', f1_scores, sep='\n')

    # uncomment block below for prompt-wise evaluation
    '''
    accuracy_scores, f1_scores = get_prompt_wise_scores(ANNOTATIONS, CHAT_GPT)

    print(accuracy_scores)
    print('ACC', sum(accuracy_scores)/(len(accuracy_scores)))
    print(f1_scores)
    print('F1', sum(f1_scores)/(len(f1_scores)))
    '''