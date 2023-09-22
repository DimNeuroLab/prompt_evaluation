import sys
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import openai
import signal
from utils import get_api_key
import threading

FEATURES = pd.read_csv('data/features.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/annotations.tsv', sep='\t')

openai.api_key = get_api_key()
model_name =   'text-davinci-003' # "gpt-3.5-turbo" #


def get_positive_few_shot_example(feature_name, prompt, shots=1):
    relevant = ANNOTATIONS[['prompt', feature_name]]
    try:
        rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 1)]
        return rel_rows.sample(shots)['prompt'].values
    except:
        return ''


def get_negative_few_shot_example(feature_name, prompt, shots=1):
    relevant = ANNOTATIONS[['prompt', feature_name]]
    try:
        rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 0)]
        return rel_rows.sample(shots)['prompt'].values
    except:
        return ''


class timeoutLinux:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class timeoutWindows:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = 100.0
        self.error_message = error_message

    def handle_timeout(self):
        print("timemout")
        raise TimeoutError(self.error_message)

    def __enter__(self):
        #signal.signal(signal.SIGALRM, self.handle_timeout)
        #signal.alarm(self.seconds)
        print("seconds",self.seconds)
        self.timer=threading.Timer(self.seconds,self.handle_timeout)
        self.timer.start()

    def __exit__(self, type, value, traceback):
        self.timer.cancel()
        #signal.alarm(0)


def createPrompt(eval_prompt, feature,shots):
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
    # include = FEATURES.loc[FEATURES['feature_name'] == feature]['include'].iloc[0]
    positive_few_shot1 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)
    # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
    positive_few_shot1 = '\n'.join(positive_few_shot1)
    negative_few_shot1 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)
    # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
    negative_few_shot1 = '\n'.join(negative_few_shot1)

    positive_few_shot2 = get_positive_few_shot_example(feature, eval_prompt, shots=shots)
    # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
    positive_few_shot2 = '\n'.join(positive_few_shot2)
    negative_few_shot2 = get_negative_few_shot_example(feature, eval_prompt, shots=shots)
    # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
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
    return eval_string,feature_description
def evaluate_prompt_logits(eval_prompt, debug=True, shots=1):
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations = []
    prompt_annotations.append(eval_prompt)

    for feature in feature_list:
        eval_string,feature_description = createPrompt(eval_prompt,feature,shots)
        conversation = [{'role': 'system', 'content': eval_string}]
        print('*'*15 + "  eval string  "+'*'*15)
        print(eval_string)
        response = None
        if debug:
            print(50*'*', feature)
            print(eval_string)
        #    response = {feature_description: -1}
        else:
            max_attempts = 5
            for _ in range(max_attempts):
                try:
                    with timeoutWindows(seconds=20):
                        '''
                        response = openai.ChatCompletion.create(
                            model=model_name,
                            messages=conversation,
                            temperature=0,
                            logit_bias={
                                15: 100.0,  # 0
                                16: 100.0,  # 1
                            }
                        )
                        '''
                        '''
                                                    import tiktoke
                                                    tokenizer = tiktoken.encoding_for_model("text-davinci-003")
                                                    tokens = [" Yes", " No"]
                                                    ids = [tokenizer.encode(token) for token in tokens]
                                                    ids
                                                    Out[6]: [[3363], [1400]]
                                                    '''
                        response = openai.Completion.create(
                            model=model_name, #'text-davinci-003',
                            prompt=eval_string,
                            max_tokens=1,
                            temperature=0,
                            logprobs=2,

                            #logit_bias={
                                # 15: 100.0,  # 0
                                # 16: 100.0,  # 1
                                #15285: 100.0,   #YES

                                #43335: 100.0,    # YES
                                #15285: 100.0,    #NO
                                #8005: 100.0     # NO
                             #   3363: 1, # " Yes"
                             #   1400: 1 # " No"

                            #},

                        )
                        '''
                        2 runs no logit bias
                        
                        "Yes": -0.12884715,
                        " Yes": -2.1852436
                        "Yes": -0.42685947,
                        " Yes": -1.0789018
                        
                        
                        '''
                        YES_string_set={"Yes","YES","Y"," Yes"," YES"," Y","Yes ","YES ","Y "," Yes "," YES "," Y ",}
                        NO_string_set = {"No", "NO", "N", " No", " NO", " N", "No ", "NO ", "N ", " No ", " NO ",
                                      " N", }

                        if (response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set or response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set):
                            break
                except Exception as EXX:
                    print("exx")
                    print(EXX)
                    print('Timeout, retrying...')
                    pass


        if response is not None and (response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set or response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set):
            print('*' * 15 + "  response  " + '*' * 15)
            #print("**** response ****")
            print(response)
            print('*' * 15 + "  response  log probs " + '*' * 15)
            value=response['choices'][0]["logprobs"]["token_logprobs"][0]
            if response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set:
                response_value_y =value
                response_value_N = -100
            else:
                response_value_y = -100
                response_value_N = value
            print(response_value_y,response_value_N)
            # response = json.loads(response['choices'][0]['message']['content'])
            #sys.exit(0)
        else:
            response_value_y = -100
            response_value_N = -100

        prompt_annotations.append(response_value_y)
        prompt_annotations.append(response_value_N)


    return prompt_annotations

from itertools import product
if __name__ == '__main__':
    df_column_names_1 = [ b+a for a, b in product(["_Y","_N"], list(ANNOTATIONS.columns)[1:])]
    print(df_column_names_1)
    df_column_names=[list(ANNOTATIONS.columns)[0]]
    df_column_names.extend(df_column_names_1)
    print(list(ANNOTATIONS.columns))
    print(df_column_names)
    df_values = []

    prompts = ANNOTATIONS['prompt'].tolist()
    for prompt in tqdm(prompts):
        # set debug=False to do actual API calls
        prompt_annotations = evaluate_prompt_logits(prompt, debug=False, shots=1)
        df_values.append(prompt_annotations)

    import time

    timestr = time.strftime("%Y%m%d-%H%M%S")
    result_data = pd.DataFrame(np.array(df_values), columns=df_column_names)
    result_data.to_csv('output/chatgpt_evaluation_log_2shots'+timestr+'.tsv', sep='\t', index=False)
