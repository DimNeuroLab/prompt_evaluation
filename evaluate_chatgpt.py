import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import glob


def compute_stats(group):
    return pd.Series({
        'Mean': group.mean(),
        'Median': group.median(),
        'Max': group.max()
    })

def models_grouped_stats(df, name,column='run', keyword = '2023'):
    # Define the keyword to split groups

    # Extract the prefix before the keyword
    df['Prefix'] = df[column].str.split(keyword, expand=True)[0]

    # Group by the 'Prefix' column
    grouped = df.groupby('Prefix')
    print("grouped")
    print(grouped)

    # Compute statistics for each group and each column
    result = grouped.apply(compute_stats).reset_index()

    # Drop the 'Prefix' column
    #result = result.drop(columns='Prefix')


    result.to_csv('output\group_eval_det_runs_' + name + '.tsv',
                           sep='\t', index=False)

def group(df,name,keyword="2023",index_column='run'):
    def extract_prefix(index):
        return index.split(keyword)[0]
    df.set_index(index_column, inplace=True)
    # Group the rows by prefix
    groups = df.groupby(extract_prefix)
    results = pd.DataFrame()

    for group_name, group_data in groups:
        # Compute the mean for each column
        group_mean ={}
        group_mean['Group'] = f'Mean_{group_name}'  # Add a new 'Group' row
        group_stdev = {}
        group_stdev['Group'] = f'StDev_{group_name}'  # Add a new 'Group' row
        group_mean.update( group_data.mean())
        group_stdev.update(group_data.std())
        # Append the group_mean to the result dataframe
        results = results.append(group_mean, ignore_index=True)
        #results = results.append(group_stdev, ignore_index=True)



    # Print the results
    print(results)
    results.to_csv('output\group_eval_det_runs_' + name + '.tsv',
                           sep='\t', index=False)
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
    pred_selection_string = "*gpt-4*" #"evaluation_detgpt-3.5*" #"pred_ense_majority_*"
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

    result_data_f1 = pd.DataFrame(features_results_f1)
    result_data_f1.to_csv('output\eval_det_runs_f1'+pred_selection_string.replace('*',"OO")+'.tsv',
                       sep='\t', index=False)

    group(result_data_f1,"_f1"+pred_selection_string.replace('*',"OO"))

    result_data_acc = pd.DataFrame(features_results_accuracy)
    result_data_acc.to_csv('output\eval_det_runs_acc_'+pred_selection_string.replace('*',"OO")+'.tsv',
                       sep='\t', index=False)
    group(result_data_acc, "_acc" + pred_selection_string.replace('*', "OO"),keyword="annotation")





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