import json

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from log_prob_callSpecificFeature import get_positive_few_shot_example, get_negative_few_shot_example
from utils import get_api_key

FEATURES = pd.read_csv('data/features.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')

openai.api_key = get_api_key()
MODEL_NAME = "gpt-3.5-turbo"


def evaluate_prompt(eval_prompt, debug=True, shots=1):
    """
    Evaluates the prompt against several features and makes annotations based
    on these evaluations.

    Args:
        eval_prompt (str): The evaluation prompt
        debug (bool, optional): Whether or not to debug the evaluation. Default is True.
        shots (int, optional): Number of prompts. Default is 1.

    Returns:
        list: List of evaluations made based on the various features.
    """
    # Extract feature list
    feature_list = FEATURES['feature_name'].tolist()

    # Initialising the prompt_annotations with given eval_prompt
    prompt_annotations = [eval_prompt]

    # Iterating over each feature and making evaluations
    for feature in feature_list:
        feature_description, _ = get_feature_description_and_include(feature)
        conversation = construct_conversation(eval_prompt, feature, feature_description, shots)

        # Debugging or getting response from AI
        response = debug_and_get_response(debug, feature, feature_description, conversation)

        response_value = validate_response(response)
        prompt_annotations.append(response_value)

    return prompt_annotations


def get_feature_description_and_include(feature):
    """
    Helper function to get feature description and include value for a given feature.
    """
    feature_info = FEATURES.loc[FEATURES['feature_name'] == feature]
    feature_description = feature_info['prompt_command'].iloc[0]
    include = feature_info['include'].iloc[0]
    return feature_description, include


def construct_conversation(eval_prompt, feature, feature_description, shots):
    """
    Helper function to construct the conversation based on the feature and eval_prompt.

    Args:
        eval_prompt (str): The evaluation prompt
        feature (str): The feature name
        feature_description (str): The description of the feature
        shots (int): Number of prompts

    Returns:
        list: conversation
    """
    # Formatting the positive and negative few shot examples
    positive_few_shot = format_few_shot_examples(get_positive_few_shot_example(feature, eval_prompt, shots))
    negative_few_shot = format_few_shot_examples(get_negative_few_shot_example(feature, eval_prompt, shots))

    eval_string = formulate_evaluation_string(feature_description, positive_few_shot, negative_few_shot, eval_prompt)

    return [{'role': 'system', 'content': eval_string}]


def format_few_shot_examples(few_shot_examples):
    """
    Helper function to format the few shot examples.
    """
    return '\n'.join(['Prompt {}: {}'.format(idx + 1, val) for idx, val in enumerate(few_shot_examples)])


def formulate_evaluation_string(feature_desc, pos_shots, neg_shots, prompt):
    """
    Helper function to formulate the evaluation string that acts as a message in the conversation.
    """
    return f"""Given the following feature:
    {feature_desc}\n
    The feature is present in the following prompts:
    {pos_shots}\n
    The feature is not present in the following prompts:
    {neg_shots}\n
    Tell me whether the feature is present in the prompt given below. Formalize your output as a json object, where the key is the feature description and the associated value is 1 if the feature is present or 0 if not.\n
    Prompt:
    {prompt}"""


def debug_and_get_response(debug, feature, feature_desc, conversation):
    """
    Helper function to choose between debugging and taking a chat AI response.
    """
    if debug:
        print_debug_info(feature, conversation[-1]['content'])
        return {feature_desc: -1}
    else:
        return get_response(conversation, feature_desc)


def print_debug_info(feature, conversation_content):
    """
    Helper function to print debug info.
    """
    print(50 * '*', feature)
    print(conversation_content)


def get_response(conversation, feature_desc):
    """
    Helper function to get response from openai Chat AI.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=conversation
        )
        return json.loads(response['choices'][0]['message']['content'])
    except:
        print(conversation[-1]['content'])
        return {feature_desc: -1}


def validate_response(response):
    """
    Helper function to validate response and assign a default value if the validation fails
    """
    try:
        return int(response[list(response.keys())[0]])
    except:
        print(response)
        return -1


if __name__ == "__main__":
    df_column_names = list(ANNOTATIONS.columns)
    df_values = []

    prompts = ANNOTATIONS['prompt'].tolist()
    for prompt in tqdm(prompts):
        # set debug=False to do actual API calls
        prompt_annotations = evaluate_prompt(prompt, debug=False, shots=2)
        df_values.append(prompt_annotations)

    result_data = pd.DataFrame(np.array(df_values), columns=df_column_names)
    result_data.to_csv('output/chatgpt_evaluation_2shots.tsv', sep='\t', index=False)
