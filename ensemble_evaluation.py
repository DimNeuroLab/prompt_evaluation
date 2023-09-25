import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score


def get_votes_for_run(true_df, pred_df):
    votes = {}
    for feature in list(true_df.columns)[1:]:
        y_pred = pred_df[feature + '_Y'].to_numpy()
        n_pred = pred_df[feature + '_N'].to_numpy()
        f_pred = np.stack((n_pred, y_pred))
        f_pred = f_pred.argmax(0)
        votes[feature] = f_pred
    return votes


def get_feature_majority_votes(true_df, *votes):
    assert len(votes) % 2 != 0
    majority_features = {}
    for feature in list(true_df.columns)[1:]:
        majority_list = []
        for v in votes:
            majority_list.append(v[feature])
        majority_vote = np.squeeze(mode(np.stack(majority_list, axis=1), axis=1)[0])
        majority_features[feature] = majority_vote
    return majority_features


def get_scores(true_df, majority_votes_dict, run, num_votes):
    accuracy_scores = {}
    f1_scores = {}
    accuracy_scores['run'] = run
    f1_scores['run'] = run
    accuracy_scores['num_votes'] = str(num_votes)
    f1_scores['num_votes'] = str(num_votes)
    for feature in list(true_df.columns)[1:]:
        y_true = true_df[feature].tolist()
        accuracy_scores[feature] = accuracy_score(y_true, majority_votes_dict[feature])
        f1_scores[feature] = f1_score(y_true, majority_votes_dict[feature], average='macro')
    return accuracy_scores, f1_scores


if __name__ == '__main__':
    ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')

    model_name = 'gpt-3.5-turbo-instruct'
    shots = '2'
    prompt_creator = '2'
    features_filename = 'new'
    annotation_filename = 'new'

    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_2_'
    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_2_features_file_features_new_'
    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_2_features_file_features_new_annotation_file_new_majority_annotations'
    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_3_'
    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_2shots_promptgen_3_features_file_features_new_annotation_file_new_majority_annotations'
    # relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_shots_3promptgen_3_features_file_features_new_annotation_file_new_majority_annotations'
    relevant_files_str = 'gpt-3.5-turbo-instruct_evaluation_log_shots_3promptgen_4_features_file_features_new_annotation_file_new_majority_annotations'

    all_files = os.listdir('output')
    all_files = [f for f in all_files if (f[:len(relevant_files_str)] == relevant_files_str and len(f) <= len(relevant_files_str) + 20)]

    #print(len(all_files))
    #sys.exit(0)

    features_results_accuracy = []
    features_results_f1 = []

    for num_votes in range(3, 12, 2):
        print('VOTES', num_votes)

        vote_files = all_files[:num_votes]
        v_list = []
        for f in vote_files:
            CHAT_GPT = pd.read_csv('output/' + f, sep='\t')
            votes = get_votes_for_run(ANNOTATIONS, CHAT_GPT)
            v_list.append(votes)

        majority_dict = get_feature_majority_votes(ANNOTATIONS, *v_list)

        acc, f1 = get_scores(ANNOTATIONS, majority_dict, relevant_files_str, num_votes)
        features_results_accuracy.append(acc)
        features_results_f1.append(f1)

    result_data = pd.DataFrame(features_results_f1)
    result_data.to_csv(f'output/ensemble_f1.tsv',
                       sep='\t', index=True)
    result_data = pd.DataFrame(features_results_accuracy)
    result_data.to_csv(f'output/ensemble_acc.tsv',
                       sep='\t', index=True)
