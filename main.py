import sys
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import openai


FEATURES = pd.read_csv('data/features.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')

openai.api_key = '' # add key here
model_name = "gpt-3.5-turbo"


def get_positive_few_shot_example(feature_name, prompt):
    relevant = ANNOTATIONS[['prompt', feature_name]]
    try:
        rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 1)]
        return rel_rows.sample(1)['prompt'].iloc[0]
    except:
        return ''


def get_negative_few_shot_example(feature_name, prompt):
    relevant = ANNOTATIONS[['prompt', feature_name]]
    try:
        rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 0)]
        return rel_rows.sample(1)['prompt'].iloc[0]
    except:
        return ''


def evaluate_prompt(eval_prompt, debug=True):
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations = []
    prompt_annotations.append(eval_prompt)

    for feature in feature_list:
        feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = FEATURES.loc[FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot = get_positive_few_shot_example(feature, eval_prompt)
        negative_few_shot = get_negative_few_shot_example(feature, eval_prompt)

        eval_string = f"""Given the following feature:
        {feature_description}\n
        The feature is present in the following prompt:
        {positive_few_shot}\n
        The feature is not present in the following prompt:
        {negative_few_shot}\n
        Tell me whether the feature is present in the prompt given below. Formalize your output as a json object, where the key is the feature description and the associated value is 1 if the feature is present or 0 if not.\n
        Prompt:
        {eval_prompt}"""
        conversation = [{'role': 'system', 'content': eval_string}]

        if debug:
            print(50*'*', feature)
            print(eval_string)
            response = {feature_description: -1}
        else:
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=conversation
                )
                response = json.loads(response['choices'][0]['message']['content'])
            except:
                print(eval_string)
                response = {feature_description: -1}

        try:
            key = list(response.keys())[0]
            response_value = int(response[key])
        except:
            print(response)
            response_value = -1
        prompt_annotations.append(response_value)

    return prompt_annotations


if __name__ == '__main__':

    df_column_names = list(ANNOTATIONS.columns)
    df_values = []

    prompts = ANNOTATIONS['prompt'].tolist()
    for prompt in tqdm(prompts):
        # set debug=False to do actual API calls
        prompt_annotations = evaluate_prompt(prompt, debug=False)
        df_values.append(prompt_annotations)

    result_data = pd.DataFrame(np.array(df_values), columns=df_column_names)
    result_data.to_csv('chatgpt_evaluation.tsv', sep='\t', index=False)
