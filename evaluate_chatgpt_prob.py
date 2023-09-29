import glob

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# def get_predicted_scores(predictions_df, feature):
#     # Create a boolean mask for filtering
#     mask = (predictions_df["feature_name"] == feature)
#
#     # Apply the mask to filter rows
#     filtered_df = predictions_df[mask]
#
#     if "class" in predictions_df.columns:
#         # Extract the classification for the det model
#         classification = filtered_df["class"].values[0]
#     else:
#         # Extract the classification for the prob model
#         value_Y = filtered_df["Y"].values[0]
#         value_N = filtered_df["N"].values[0]
#
#         if value_Y > value_N:
#             classification = 1
#         elif value_N > value_Y:
#             classification = 0
#         else:
#             classification = -1
#
#     return classification
#
#
# def get_prompt_wise_scores(true_df, pred_df, filler=0):
#     accuracy_scores = []
#     f1_scores = []
#
#     for idx, row in true_df.iterrows():
#         y_true = row.values[1:]
#
#         y_pred = pred_df.iloc[idx].values[1:]
#         y_pred[y_pred == -1] = filler
#
#         y_true = list(y_true)
#         y_pred = list(y_pred)
#
#         accuracy_scores.append(accuracy_score(y_true, y_pred))
#         f1_scores.append(f1_score(y_true, y_pred, average="macro"))
#
#     return accuracy_scores, f1_scores
#
#
# def get_feature_wise_scores(true_df, pred_df, filename):
#     accuracy_scores = {}
#     f1_scores = {}
#
#     accuracy_scores['run'] = filename
#     f1_scores['run'] = filename
#
#     for feature in list(true_df.columns)[1:]:
#         y_true = true_df[feature].tolist()
#
#         y_pred = pred_df[feature + '_Y'].to_numpy()
#         n_pred = pred_df[feature + '_N'].to_numpy()
#
#         f_pred = np.stack([n_pred, y_pred])
#         f_pred = f_pred.argmax(0)
#
#         accuracy_scores[feature] = accuracy_score(y_true, f_pred)
#         f1_scores[feature] = f1_score(y_true, f_pred, average='macro')
#
#     return accuracy_scores, f1_scores


def calculate_feature_metrics(predictions_df):
    accuracy_scores = dict()
    f1_scores = dict()

    features = predictions_df["feature_name"].unique()

    for feature in features:
        mask = (predictions_df["feature_name"] == feature)

        # Apply the mask to filter rows
        filtered_df = predictions_df[mask]

        ground_truths = predictions_df["gt"].tolist()

        if "class" in predictions_df.columns:
            # Extract the classification for the det model
            classifications = filtered_df["class"].tolist()
        else:
            # Extract the classification for the prob model
            is_true = (filtered_df["Y"] > filtered_df["N"]).tolist()
            # is_false = filtered_df["Y"] < filtered_df["N"]
            # is_equal = filtered_df["Y"] == filtered_df["N"]

            classifications = [
                int(val) for val in is_true
            ]

        accuracy_scores[feature] = accuracy_score(ground_truths, classifications)
        f1_scores[feature] = f1_score(ground_truths, classifications, average='macro')

    return accuracy_scores, f1_scores


def evaluate_prob(annotations, files):
    features_results_accuracy = []
    features_results_f1 = []

    for f in files:
        CHAT_GPT = pd.read_csv(f, sep='\t')

        # block below for feature-wise evaluation
        # accuracy_scores, f1_scores = get_feature_wise_scores(annotations, CHAT_GPT, f)
        accuracy_scores, f1_scores = calculate_feature_metrics(CHAT_GPT)

        print('ACC', accuracy_scores, sep='\n')
        print('F1', f1_scores, sep='\n')

        features_results_accuracy.append(accuracy_scores)
        features_results_f1.append(f1_scores)

    result_data = pd.DataFrame(features_results_f1)
    result_data.to_csv(
        'output/evaluation_prob_runs_f1.tsv',
        sep='\t', index=False
    )

    result_data = pd.DataFrame(features_results_accuracy)
    print(result_data)
    result_data.to_csv(
        'output/evaluation_prob_runs_accuracy.tsv',
        sep='\t', index=False
    )

    # uncomment block below for prompt-wise evaluation
    '''
        accuracy_scores, f1_scores = get_prompt_wise_scores(ANNOTATIONS, CHAT_GPT)

        print(accuracy_scores)
        print('ACC', sum(accuracy_scores)/(len(accuracy_scores)))
        print(f1_scores)
        print('F1', sum(f1_scores)/(len(f1_scores)))
    '''


if __name__ == '__main__':
    ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')
    files = glob.glob("output/single_feature/gpt-3.5-turbo-instruct*.tsv")

    print(files)

    evaluate_prob(ANNOTATIONS, files)
