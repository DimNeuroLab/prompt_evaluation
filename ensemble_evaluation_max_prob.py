import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score
import glob
import time


def get_probs_for_run(columns, pred_df):
    #list_y={pred_df.columns[0]:columns[0]}
    list_y={feature + '_Y':feature for feature in columns}
    #list_n ={pred_df.columns[0]:columns[0]}
    list_n={feature + '_N':feature for feature in columns}
    #y_pred=pred_df.filter(items=list_y.keys())
    #n_pred =pred_df.filter(items=list_n.keys())
    y_pred =  pred_df[list_y.keys()].copy()
    n_pred = pred_df[list_n.keys()].copy()

    y_pred.rename(columns=list_y,inplace=True)
    n_pred.rename(columns=list_n,inplace=True)

    return y_pred,n_pred


def all_aggregations(pd_list,threshold=-99):
    # Create empty dataframes to store max, argmax, sum, and median values
    max_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)
    argmax_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)
    sum_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)
    median_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)
    count_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)
    avg_thresh = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)

    # Iterate through cells and compute max, argmax, sum, and median
    for col in pd_list[0].columns:
        for row in pd_list[0].index:
            cell_values = [df.at[row, col] for df in pd_list]
            max_df.at[row, col] = max(cell_values)
            argmax_df.at[row, col] = cell_values.index(max(cell_values))
            sum_df.at[row, col] = sum(cell_values)
            median_df.at[row, col] = np.median(cell_values)
            count_df.at[row,col]=sum(value > threshold for value in cell_values)
            avg_thresh.at[row,col]=threshold
            if count_df.at[row, col]>0:
                avg_thresh.at[row,col]=sum(value  for value in cell_values if value> threshold)/count_df.at[row,col]
    return {'max':max_df,'argmax':argmax_df,'sum':sum_df,'median':median_df,'count':count_df,'avg_thresh':avg_thresh}

def argmax(pd_list):
    # Create empty dataframes to store max, argmax, sum, and median values
    argmax_df = pd.DataFrame(columns=pd_list[0].columns, index=pd_list[0].index)

    # Iterate through cells and compute max, argmax, sum, and median
    for col in pd_list[0].columns:
        for row in pd_list[0].index:
            cell_values = [df.at[row, col] for df in pd_list]
            argmax_df.at[row, col] = cell_values.index(max(cell_values))
    return argmax_df

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
    # Set the first row as column names
    #ANNOTATIONS.columns = ANNOTATIONS.iloc[0]

    # Set the first column as the index
    ANNOTATIONS = ANNOTATIONS.set_index(ANNOTATIONS.columns[0])

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

    #all_files = os.listdir('output')
    #all_files = [f for f in all_files if (f[:len(relevant_files_str)] == relevant_files_str and len(f) <= len(relevant_files_str) + 20)]
    pred_selection_string="evaluation_prob_gpt-*_shots_2promptgen_*" #"evaluation_prob_gpt*"
    all_files = glob.glob("output/"+pred_selection_string+".tsv")
    #print(len(all_files))
    #sys.exit(0)

    features_results_accuracy = []
    features_results_f1 = []

    for num_votes in range(9, 21, 2):
        print('VOTES', num_votes)

        for permutation in range(5):
            import random

            shuffled_list = list(all_files)

            # Shuffle the list to get a random permutation
            random.shuffle(shuffled_list)
            vote_files = shuffled_list[:num_votes]
            print("vote_files")
            print(vote_files)
            p_list_y = []
            p_list_n=[]
            for f in vote_files:
                CHAT_GPT = pd.read_csv(f, sep='\t')
                #CHAT_GPT.columns = CHAT_GPT.iloc[0]

                # Set the first column as the index
                CHAT_GPT = CHAT_GPT.set_index(CHAT_GPT.columns[0])

                probs_y,probs_n = get_probs_for_run(ANNOTATIONS.columns, CHAT_GPT)
                p_list_y.append(probs_y)
                p_list_n.append(probs_n)
            y_stats=all_aggregations(p_list_y)
            n_stats=all_aggregations(p_list_n)
            majority_counts=argmax([n_stats['count'],y_stats['count']])
            majority_values = argmax([n_stats['max'], y_stats['max']])
            majority_avg_thresh = argmax([n_stats['avg_thresh'], y_stats['avg_thresh']])

            # p_list_y=pd.concat(p_list_y,axis=2)
            # p_list_n = pd.concat(p_list_n,keys=vote_files)
            # majority_dict = get_feature_highest_prob_vote(ANNOTATIONS, p_list_y,p_list_n)

            timestr = time.strftime("%Y%m%d-%H%M%S")
            majority_counts.to_csv('output/pred_ense_majority_counts_sel_str'+pred_selection_string.replace('*', 'OO')+
                                   "_votes_"+str(num_votes)+"_annotation_file_"
                               + annotation_filename + '_' + timestr + 'perm'+str(permutation)+'.tsv', sep='\t')
            majority_values.to_csv('output/pred_ense_majority_max_sel_str'+pred_selection_string.replace('*', 'OO') +
                                   "_votes_" +str(num_votes)+ "_annotation_file_"
                                   + annotation_filename + '_' + timestr + 'perm'+str(permutation)+'.tsv', sep='\t')
            majority_avg_thresh.to_csv('output/pred_ense_majority_avg_thresh_sel_str'+pred_selection_string.replace('*', 'OO')+
                                       "_votes_" +str(num_votes)+ "_annotation_file_"
                                   + annotation_filename + '_' + timestr + 'perm'+str(permutation)+'.tsv', sep='\t')


