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
from promptCreators import promptCreator as pc
import functools

features_filename = 'features_new'
annotation_filename = 'new_majority_annotations'
FEATURES = pd.read_csv('data/'+features_filename+'.tsv', sep='\t')
ANNOTATIONS = pd.read_csv('data/'+annotation_filename+'.tsv', sep='\t')
ANNOTATIONS_TEST=ANNOTATIONS
the_feat = "1 Goal (1,NaN)"
feature_list = FEATURES['feature_name'].tolist()
#feature_list = [the_feat] #FEATURES['feature_name'].tolist()
openai.api_key = get_api_key()
model_name_det =   "gpt-3.5-turbo"  #"gpt-4"
model_name_prob =   "gpt-3.5-turbo-instruct" #'text-davinci-003'
promptCreator_ids=[6]
shots=2
num_runs= 3

eval_det = True
eval_prob = True
YES_string_set={"Yes","YES","Y"," Yes"," YES"," Y","Yes ","YES ","Y "," Yes "," YES "," Y ",}
NO_string_set = {"No", "NO", "N", " No", " NO", " N", "No ", "NO ", "N ", " No ", " NO "," N", }




@retry(wait=wait_random_exponential(min=1, max=240), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=240), stop=stop_after_attempt(4))
def chatcompletion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def get_true_label(feature_name, prompt, shots=1):
    relevant = ANNOTATIONS_TEST[['prompt', feature_name]]
    try:
        rel_rows = relevant.loc[(relevant['prompt'] == prompt)]
        return rel_rows.sample(shots)[feature_name].values
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
    def __init__(self, seconds=1, error_message='Timeout '):
        self.seconds = seconds
        self.error_message = error_message +' '+str(seconds)

    def handle_timeout(self):
        print("timemout thrown")
        raise TimeoutError(self.error_message)

    def __enter__(self):
        #signal.signal(signal.SIGALRM, self.handle_timeout)
        #signal.alarm(self.seconds)
        print("seconds before timeout: ",self.seconds)
        self.timer=threading.Timer(self.seconds,self.handle_timeout)
        self.timer.start()

    def __exit__(self, type, value, traceback):
        self.timer.cancel()
        #signal.alarm(0)


# def timeout(func):
#     def inner_func(*nums, **kwargs):
#         t = threading.Thread(target=func, args=(*nums,))
#         t.start()
#         t.join(timeout=5)
#     return inner_func

def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, seconds_before_timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = threading.Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print('error starting thread')
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

def evaluate_prompt_both( featurelist,eval_prompt, shots, promptCreator, debug=True):
    import copy
    prompt_annotations_det = {}
    prompt_annotations_det['eval_prompt'] = eval_prompt
    prompt_annotations_prob=dict(prompt_annotations_det)


    for feature in featurelist:
        eval_string, feature_description = promptCreator.getPrompt(eval_prompt, feature, shots)


        conversation = [{'role': 'system', 'content': eval_string}]
        print('*' * 15 + "  eval string  " + '*' * 15)
        print(eval_string)

        print("ground truth")

        print(get_true_label(feature, prompt))
        eval_string_prob = eval_string
        eval_string_det = eval_string
        if eval_prob:
            prompt_annotations_prob.update(evaluate_prompt_logits(feature,eval_string_prob, eval_prompt,debug=False))
        if eval_det:
            prompt_annotations_det.update(evaluate_prompt_det(feature,eval_string_det, feature_description,conversation,eval_prompt,debug=False))

    return prompt_annotations_prob, prompt_annotations_det
def evaluate_prompt_logits(feature,eval_string,  eval_prompt, debug=True):
    #these must be defined in the outer loop in orde to integrate all the feature for all the prompts
    prompt_annotations = {}
    prompt_annotations['eval_prompt']=eval_prompt


    response = None
    if debug:
        print(50*'*', feature)
        #print(eval_string)
    #    response = {feature_description: -1}
    else:
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                time.sleep(0.15)
                with timeoutWindows(seconds=5):
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
                        model=model_name_prob, #'text-davinci-003',
                        prompt=eval_string,
                        max_tokens=1,
                        temperature=0,
                        logprobs=2,
                        #messages=conversation,

                        logit_bias={
                            # 15: 100.0,  # 0
                            # 16: 100.0,  # 1
                            #15285: 100.0,   #YES

                            #43335: 100.0,    # YES
                            #15285: 100.0,    #NO
                            #8005: 100.0     # NO
                            #3363: 1, # " Yes"
                            #1400: 1 # " No"

                        },

                    )
                    '''
                    2 runs no logit bias
                    
                    "Yes": -0.12884715,
                    " Yes": -2.1852436
                    "Yes": -0.42685947,
                    " Yes": -1.0789018
                    
                    
                    '''



                    if (response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set or response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set):
                        break
                    else:
                        print('+' * 80)
                        print("bad response")
                        print(response)
                        print('+'*80)
            except Exception as EXX:
                print("exx")
                print(EXX)
                print('Timeout, retrying...')
                pass


    if response is not None and (response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set or response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set):
        gt = get_true_label(feature, prompt)
        if response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set:
            print("\n\n*************** RESPONSE PROB YES ****************\n\n")
            if gt[0]!=1:
                print("MESSMESSMESSMESSMESSMESSMESSMESSMESSMESS")
        elif response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set:

            print("\n\n*************** RESPONSE PROB NO ****************\n\n" )
            if gt[0]!=0:
                print("MESSMESSMESSMESSMESSMESSMESSMESSMESSMESS")
        else:
            print("\n\n*************** RESPONSE PROB UFO ****************\n\n")
            print("MESSMESSMESSMESSMESSMESSMESSMESSMESSMESS")
        print('*' * 15 + " full  response prob  " + '*' * 15)
        #print("**** response ****")
        print(response)

        value=response['choices'][0]["logprobs"]["token_logprobs"][0]
        if response['choices'][0]["logprobs"]["tokens"][0] in YES_string_set:
            response_value_Y =value
            response_value_N = -100
        elif response['choices'][0]["logprobs"]["tokens"][0] in NO_string_set:
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
    prompt_annotations[feature+'_Y']=response_value_Y
    prompt_annotations[feature+'_N']=response_value_N

    return prompt_annotations

def evaluate_prompt_det( feature,eval_string,feature_description, conversation,eval_prompt, debug=True):
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations={}
    prompt_annotations["eval_prompt"]=eval_prompt
    print("DET")
    if debug:
        print(50*'*', feature)
        #print(eval_string)
        response = {feature_description: -1}
    else:
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                time.sleep(0.15)
                #with timeoutWindows(seconds=30):


                response =timeout(5)(openai.ChatCompletion.create)(
                    model=model_name_det,
                    messages=conversation
                )
                print("DETERMINISTIC RESPONSE1")
                print("DETERMINISTIC RESPONSE2")
                print("DETERMINISTIC RESPONSE3")
                print("full response det parsed")
                print(response)
                response_parsed=response['choices'][0]['message']['content']
                print("response det parsed")
                print(response_parsed)

                if response_parsed in YES_string_set:
                        response_value =1
                        break
                elif response_parsed in NO_string_set:
                        response_value=-1
                        break
                else:
                        print("parse error")
                        response_value=0
                        # except:
                        #     print("except response_parsed")
                        #     print(response_parsed)
                        #     response_value = -1

            except Exception as EXX:
                print("exx")
                print(EXX)
                print("DETERMINISTIC RESPONSE14")
                print("DETERMINISTIC RESPONSE25")
                print("DETERMINISTIC RESPONSE36")
                #print(eval_string)
                response_parsed = -1
                pass



    prompt_annotations[feature]=response_value

    return prompt_annotations

from itertools import product
if __name__ == '__main__':
    global not_good_response
    not_good_response = 0
    #df_column_names_1 = [ b+a for a, b in product(["_Y","_N"], list(ANNOTATIONS.columns)[1:])]
    df_column_names_1 = [ b+a for a, b in product(["_Y","_N"],[the_feat])]
    print(df_column_names_1)
    df_column_names=[list(ANNOTATIONS.columns)[0]]
    df_column_names.extend(df_column_names_1)
    print(list(ANNOTATIONS.columns))
    print(df_column_names)
    for promptCreator_id in promptCreator_ids:
        promptCreator=pc(FEATURES,ANNOTATIONS,promptCreator_id)

        for _ in range(num_runs):
            df_values_prob = []
            df_values_det = []

            prompts = ANNOTATIONS['prompt'].tolist()

            for counter,prompt in enumerate(prompts):
                print('+'*60)
                print('+' * 60)
                print(counter,prompt)
                print('+' * 60)
                print('+' * 60)

                prompt_annotations_prob,det_annotations = evaluate_prompt_both(feature_list,prompt, debug=False, shots=shots,promptCreator=promptCreator)
                df_values_prob.append(prompt_annotations_prob)
                df_values_det.append(det_annotations)

                #if counter>1:
                  #  break
    #sathya remember to save det_annotations



            print("not good response")
            print(not_good_response)



            timestr = time.strftime("%Y%m%d-%H%M%S")
            result_data = pd.DataFrame(df_values_prob)
            result_data.to_csv('output/evaluation_prob_'+model_name_prob+'_'+model_name_det+'_shots_'+str(shots)+
                               'promptgen_'+str(promptCreator_id)+"_features_file_"+features_filename+"_annotation_file_"
                               +annotation_filename+'_'+timestr+'nobias.tsv', sep='\t', index=False)

            result_data = pd.DataFrame(df_values_det)
            result_data.to_csv('output/evaluation_det' + model_name_prob + ' ' + model_name_det + '_shots_' +
                               str(shots) + 'promptgen_' + str(promptCreator_id) + "_features_file_" + features_filename +
                               "_annotation_file_" + annotation_filename + '_' + timestr + 'nobias.tsv',
                               sep='\t', index=False)

