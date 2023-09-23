import sys
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import openai
import signal
from utils import get_api_key
import threading
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


features_filename = 'features_new'
annotation_filename  = 'new_majority_annotations'
FEATURES = pd.read_csv('data/'+features_filename+'.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/'+annotation_filename+'.tsv', sep='\t')

openai.api_key = get_api_key()
model_name =   "gpt-3.5-turbo-instruct" #'text-davinci-003' # "gpt-3.5-turbo" # #"gpt-4"
promptCreator=2
num_runs= 1



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

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

def createPromptInverted(eval_prompt, feature,shots):
    '''

    :param eval_prompt:
    :param feature:
    :param shots:
    :return:

    output example (sequence of labels used in the few shot learning: N Y Y N)
            Me: Answer with Yes or No if this feature:
            additional contextual information about the role of the language model, the user, or the environment

            is present in the following prompt:

            Explain the negative sides of social media use without using bulletins and ask one question at a time. And make it interactive by asking questions like a teacher

            You: No

            Me: and in the following prompt?

            I'm a student!  Could you be my super-cool "teacher" for a bit and chat about two tricky things with social media "Echo Chambers" and "Social Media Self Protection Skills"?  Here's how we can make it awesome:   - Let's make this a real conversation. You ask one question at a time, always hold up for my reply, I answer, and go to the next interactive step.  - Keep the conversation fun! A joke or two wouldn't hurt.  - First, what is my name, how old am I and what's my school level? That way, you can keep things more appropriate for me.  - Lastly, what's my cultural background? It'll help make our chat about social media even more understandable by mentioning related examples specific to my culture for each topic.

            You: Yes


            Me: and in the following prompt?

            Hello! Please try to act like my teacher teaching me disadvantages of social media by considering my age, level of education, and culture but in a more friendly and supportive way. Meanwhile, please do this in an interactive way by asking one question at a time.

            You: Yes

            Me: and in the following prompt?

            I want you to teach me the disadvantages of social media according to my personal information like age, level of education, & culture.

            You: No


            Me: and in the following prompt?

            Hello! I want to learn more about the negative aspects of social media. Can we have an educative conversation about it?

            You:
    '''
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
    return eval_string,feature_description


def createPromptRevised(eval_prompt, feature, shots):
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
def evaluate_prompt_logits(eval_prompt, debug=True, shots=1,promptCreator=2):
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations = []
    prompt_annotations.append(eval_prompt)

    for feature in feature_list:
        if promptCreator==1:
            eval_string,feature_description = createPrompt(eval_prompt,feature,shots)
        elif promptCreator==2:
            eval_string, feature_description = createPromptInverted(eval_prompt, feature, shots)
        elif promptCreator==3:
            eval_string, feature_description = createPromptRevised(eval_prompt, feature, shots)
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
                    time.sleep(0.5)
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
                        response = completion_with_backoff(
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
                response_value_Y =value
                response_value_N = -100
            else:
                response_value_Y = -100
                response_value_N = value

            # response = json.loads(response['choices'][0]['message']['content'])
            #sys.exit(0)
        else:
            response_value_Y = -100
            response_value_N = -100
            global not_good_response
            not_good_response += 1
        print(response_value_Y, response_value_N)
        prompt_annotations.append(response_value_Y)
        prompt_annotations.append(response_value_N)




    return prompt_annotations

from itertools import product
if __name__ == '__main__':
    global not_good_response
    not_good_response = 0
    df_column_names_1 = [ b+a for a, b in product(["_Y","_N"], list(ANNOTATIONS.columns)[1:])]
    print(df_column_names_1)
    df_column_names=[list(ANNOTATIONS.columns)[0]]
    df_column_names.extend(df_column_names_1)
    print(list(ANNOTATIONS.columns))
    print(df_column_names)
    for _ in range(num_runs):
        df_values = []

        prompts = ANNOTATIONS['prompt'].tolist()
        for prompt in tqdm(prompts):
            # set debug=False to do actual API calls
            prompt_annotations = evaluate_prompt_logits(prompt, debug=False, shots=1,promptCreator=promptCreator)
            df_values.append(prompt_annotations)


        print("not good response")
        print(not_good_response)



        timestr = time.strftime("%Y%m%d-%H%M%S")
        result_data = pd.DataFrame(np.array(df_values), columns=df_column_names)
        result_data.to_csv('output/'+model_name+'_evaluation_log_2shots_promptgen_'+str(promptCreator)+"_features_file_"+features_filename+"_annotation_file_"+annotation_filename+'_'+timestr+'.tsv', sep='\t', index=False)

