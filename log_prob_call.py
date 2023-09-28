import time

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from log_prob_callSpecificFeature import get_positive_few_shot_example, get_negative_few_shot_example, timeoutWindows, \
    completion_with_backoff, YES_STRINGS, NO_STRINGS
from utils import get_api_key

features_filename = 'features_new_revised_goals'
annotation_filename = 'new_majority_annotations'
FEATURES = pd.read_csv('data/' + features_filename + '.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/' + annotation_filename + '.tsv', sep='\t')

openai.api_key = get_api_key()
model_name = "gpt-3.5-turbo-instruct"  # 'text-davinci-003' # "gpt-3.5-turbo" # #"gpt-4"
promptCreator = 6
shots = 3
num_runs = 5


def createPrompt(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot1 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot1 = '\n'.join(positive_few_shot1)
    negative_few_shot1 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    negative_few_shot1 = '\n'.join(negative_few_shot1)

    positive_few_shot2 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot2 = '\n'.join(positive_few_shot2)
    negative_few_shot2 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)
    negative_few_shot2 = '\n'.join(negative_few_shot2)

    eval_string = f"""Me: Check if this feature:
            {feature_description}\n
            is present in the following prompts, answer with YES or NO\n
            {positive_few_shot1}\n
            You: Yes\n
            Me: and in the following prompt?
            {negative_few_shot1}\n
            You: No\n

            Me: and in the following prompt?
            {negative_few_shot2}\n
            You: No\n
            Me: and in the following prompt?
            {positive_few_shot2}\n
            You: Yes\n

            Me: and in the following prompt?
            {eval_prompt}\n
            You: \n
            """
    return eval_string, feature_description


def createPromptInverted(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot1 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot1 = '\n'.join(positive_few_shot1)
    negative_few_shot1 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    negative_few_shot1 = '\n'.join(negative_few_shot1)

    positive_few_shot2 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot2 = '\n'.join(positive_few_shot2)
    negative_few_shot2 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    negative_few_shot2 = '\n'.join(negative_few_shot2)

    eval_string = f"""Me: Answer with Yes or No if this feature:
            {feature_description}\n
            is present in the following prompt:\n
            {negative_few_shot2}\n
            You: No\n
            Me: and in the following prompt?\n
            {positive_few_shot2}\n
            You: Yes\n

            Me: and in the following prompt?\n
            {positive_few_shot1}\n
            You: Yes\n
            Me: and in the following prompt?\n
            {negative_few_shot1}\n
            You: No\n
            
            Me: and in the following prompt?\n
            {eval_prompt}\n
            You: \n
            """
    return eval_string, feature_description


def createPromptRandom(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot = []
    negative_few_shot = []
    eval_string = f"""
            Me: Answer with Yes or No if this feature:
                {feature_description}\n
            is present in the following prompt:\n"""
    for i in range(shots):
        positive_few_shot.append(get_positive_few_shot_example(feature, eval_prompt, shots=1))

        positive_few_shot[i] = '\n'.join(positive_few_shot[i])
        negative_few_shot.append(get_negative_few_shot_example(feature, eval_prompt, shots=1))

        negative_few_shot[i] = '\n'.join(negative_few_shot[i])
        if np.random.choice(2, 1) == 1:
            newsubstring = f"""
            {negative_few_shot[i]}\n
            You: No\n
            Me: and in the following prompt?\n
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: and in the following prompt?\n"""
            eval_string += newsubstring
        else:
            eval_string += f"""
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: and in the following prompt?\n
            {negative_few_shot[i]}\n
            You: No\n
            Me: and in the following prompt?\n"""

    eval_string += f"""
            {eval_prompt}\n
            You: \n
            """

    return eval_string, feature_description


def createPromptRevised(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot1 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot1 = '\n'.join(positive_few_shot1)
    negative_few_shot1 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    negative_few_shot1 = '\n'.join(negative_few_shot1)

    positive_few_shot2 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)

    positive_few_shot2 = '\n'.join(positive_few_shot2)
    negative_few_shot2 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    negative_few_shot2 = '\n'.join(negative_few_shot2)

    eval_string = f"""Me: Answer with Yes or No if this feature:
            {feature_description}\n
            is present in the following prompt:\n
            {positive_few_shot1}\n
            You: Yes\n
            Me: and in the following prompt?\n
            {negative_few_shot1}\n
            You: No\n
            Me: and in the following prompt?\n
            {negative_few_shot2}\n
            You: No\n
            Me: and in the following prompt?\n
            {positive_few_shot2}\n
            You: Yes\n

            Me: and in the following prompt?\n
            {eval_prompt}\n
            You: \n
            """
    return eval_string, feature_description


def createPromptRandom2(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot = []
    negative_few_shot = []
    eval_string = f"""
            Me: Answer with Yes or No if this feature:
                {feature_description}\n
            applies ...\n
            to the following prompt:\n"""
    for i in range(shots):
        positive_few_shot.append(get_positive_few_shot_example(feature, eval_prompt, shots=1))

        positive_few_shot[i] = '\n'.join(positive_few_shot[i])
        negative_few_shot.append(get_negative_few_shot_example(feature, eval_prompt, shots=1))

        negative_few_shot[i] = '\n'.join(negative_few_shot[i])
        if np.random.choice(2, 1) == 1:
            newsubstring = f"""
            {negative_few_shot[i]}\n
            You: No\n
            Me: to the following prompt:\n
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: to the following prompt:\n"""
            eval_string += newsubstring
        else:
            eval_string += f"""
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: to the following prompt:\n
            {negative_few_shot[i]}\n
            You: No\n
            Me: to the following prompt:\n"""

    eval_string += f"""
            {eval_prompt}\n
            You: \n
            """

    return eval_string, feature_description


def createPromptRandom3(eval_prompt, feature, shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    positive_few_shot = []
    negative_few_shot = []
    eval_string = f"""
            Me: Answer with Yes or No if this description:
                {feature_description}\n
            applies ...\n
            to the following prompt:\n"""
    for i in range(shots):
        positive_few_shot.append(get_positive_few_shot_example(feature, eval_prompt, shots=1))

        positive_few_shot[i] = '\n'.join(positive_few_shot[i])
        negative_few_shot.append(get_negative_few_shot_example(feature, eval_prompt, shots=1))

        negative_few_shot[i] = '\n'.join(negative_few_shot[i])
        if np.random.choice(2, 1) == 1:
            newsubstring = f"""
            {negative_few_shot[i]}\n
            You: No\n
            Me: to the following prompt:\n
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: to the following prompt:\n"""
            eval_string += newsubstring
        else:
            eval_string += f"""
            {positive_few_shot[i]}\n
            You: Yes\n
            Me: to the following prompt:\n
            {negative_few_shot[i]}\n
            You: No\n
            Me: to the following prompt:\n"""

    eval_string += f"""
            {eval_prompt}\n
            You: \n
            """

    return eval_string, feature_description


def evaluate_prompt_logits(eval_prompt, debug=True, shots=1, promptCreator=2):
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations = {}
    prompt_annotations['prompt'] = eval_prompt

    # Sathya be careful with this for loop it is in the opposite order of the new code
    for feature in feature_list:
        if promptCreator == 1:
            eval_string, feature_description = createPrompt(eval_prompt, feature, shots)
        elif promptCreator == 2:
            eval_string, feature_description = createPromptInverted(eval_prompt, feature, shots)
        elif promptCreator == 3:
            eval_string, feature_description = createPromptRevised(eval_prompt, feature, shots)
        elif promptCreator == 4:
            eval_string, feature_description = createPromptRandom(eval_prompt, feature, shots)
        elif promptCreator == 5:
            eval_string, feature_description = createPromptRandom2(eval_prompt, feature, shots)
        elif promptCreator == 6:
            eval_string, feature_description = createPromptRandom3(eval_prompt, feature, shots)

        conversation = [{'role': 'system', 'content': eval_string}]
        print('*' * 15 + "  eval string  " + '*' * 15)
        print(eval_string)
        response = None
        if debug:
            print(50 * '*', feature)
            print(eval_string)
        else:
            max_attempts = 5
            for _ in range(max_attempts):
                try:
                    time.sleep(0.5)
                    with timeoutWindows(seconds=100):
                        response = completion_with_backoff(
                            model=model_name,  # 'text-davinci-003',
                            prompt=eval_string,
                            max_tokens=1,
                            temperature=0,
                            logprobs=2,

                            logit_bias={},

                        )

                        if (response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS or
                                response['choices'][0]["logprobs"]["tokens"][0] in NO_STRINGS):
                            break
                        else:
                            print('+' * 80)
                            print("bad response")
                            print(response)
                            print('+' * 80)
                except Exception as EXX:
                    print("exx")
                    print(EXX)
                    print('Timeout, retrying...')
                    pass

        if response is not None and (
                response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS or
                response['choices'][0]["logprobs"]["tokens"][0] in NO_STRINGS
        ):
            print('*' * 15 + "  response  " + '*' * 15)

            print(response)
            print('*' * 15 + "  response  log probs " + '*' * 15)
            value = response['choices'][0]["logprobs"]["token_logprobs"][0]
            if response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS:
                response_value_Y = value
                response_value_N = -100
            else:
                response_value_Y = -100
                response_value_N = value
        else:
            response_value_Y = -100
            response_value_N = -100
            global not_good_response
            not_good_response += 1
        print(response_value_Y, response_value_N)
        prompt_annotations[feature + '_Y'] = response_value_Y
        prompt_annotations[feature + '_N'] = response_value_N

    return prompt_annotations


from itertools import product

if __name__ == '__main__':
    global not_good_response
    not_good_response = 0
    df_column_names_1 = [a + b for a, b in product(list(ANNOTATIONS.columns)[1:], ["_Y", "_N"])]

    print(df_column_names_1)
    df_column_names = [list(ANNOTATIONS.columns)[0]]
    df_column_names.extend(df_column_names_1)
    print(list(ANNOTATIONS.columns))
    print(df_column_names)
    for _ in range(num_runs):
        df_values = []

        prompts = ANNOTATIONS['prompt'].tolist()
        for prompt in tqdm(prompts):
            # set debug=False to do actual API calls
            prompt_annotations = evaluate_prompt_logits(prompt, debug=False, shots=shots, promptCreator=promptCreator)
            df_values.append(prompt_annotations)

        print("not good response")
        print(not_good_response)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        result_data = pd.DataFrame(df_values, columns=df_column_names)
        result_data.to_csv(
            'output/' + model_name + '_evaluation_log_shots_' + str(shots) + 'promptgen_' + str(
                promptCreator) + "_features_file_" + features_filename + "_annotation_file_" + annotation_filename + '_' + timestr + 'nobias.tsv',
            sep='\t', index=False
        )
