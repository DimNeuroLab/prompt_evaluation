import numpy as np

class promptCreator:
    def __init__(self,FEATURES,ANNOTATIONS, promptCreatorid):
        self.FEATURES=FEATURES
        self.ANNOTATIONS=ANNOTATIONS
        self.promptCreatorid=promptCreatorid
    def get_positive_few_shot_example(self, feature_name, prompt, shots=1):
        relevant = self.ANNOTATIONS[['prompt', feature_name]]
        try:
            rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 1)]
            return rel_rows.sample(shots)['prompt'].values
        except:
            return ''

    def get_negative_few_shot_example(self,feature_name, prompt, ANNOTATIONS, shots=1):
        relevant = self.ANNOTATIONS[['prompt', feature_name]]
        try:
            rel_rows = relevant.loc[(relevant['prompt'] != prompt) & (relevant[feature_name] == 0)]
            return rel_rows.sample(shots)['prompt'].values
        except:
            return ''


    def createPromptZero(self,eval_prompt, feature, shots):
        feature_description = self.FEATURES.loc[self.FEATURES['self.feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot1 = self.get_positive_few_shot_example(feature, eval_prompt,  shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot1 = '\n'.join(positive_few_shot1)
        negative_few_shot1 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
        # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
        negative_few_shot1 = '\n'.join(negative_few_shot1)

        eval_string = f"""Me: Check if this feature:
                {feature_description}\n
                is present in the following prompts, answer with YES or NO\n
                {positive_few_shot1}\n
                You: Yes\n
                Me: and in the following prompt?
                {negative_few_shot1}\n
                You: No\n
    
                Me: and in the following prompt?
                {eval_prompt}\n
                You: \n
                """
        return eval_string, feature_description


    def createPrompt(self,eval_prompt, feature, shots):
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot1 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot1 = '\n'.join(positive_few_shot1)
        negative_few_shot1 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
        # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
        negative_few_shot1 = '\n'.join(negative_few_shot1)

        positive_few_shot2 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot2 = '\n'.join(positive_few_shot2)
        negative_few_shot2 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
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
        return eval_string, feature_description


    def createPromptInverted(self,eval_prompt, feature, shots):
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
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot1 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot1 = '\n'.join(positive_few_shot1)
        negative_few_shot1 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
        # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
        negative_few_shot1 = '\n'.join(negative_few_shot1)

        positive_few_shot2 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot2 = '\n'.join(positive_few_shot2)
        negative_few_shot2 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
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
        return eval_string, feature_description


    def createPromptRandom2(self,eval_prompt, feature, shots):
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
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot = []
        negative_few_shot = []
        eval_string = f"""
                Me: Answer with Yes or No if this feature:
                    {feature_description}\n
                applies ...\n
                to the following prompt:\n"""
        for i in range(shots):
            positive_few_shot.append(self.get_positive_few_shot_example(feature, eval_prompt, shots=1))
            # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
            positive_few_shot[i] = '\n'.join(positive_few_shot[i])
            negative_few_shot.append(self.get_negative_few_shot_example(feature, eval_prompt, shots=1))
            # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
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


    def createPromptRandom3(self,eval_prompt, feature, shots):
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
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot = []
        negative_few_shot = []
        eval_string = f"""
                Me: Answer with Yes or No if this description:
                    {feature_description}\n
                applies ...\n
                to the following prompt:\n"""
        for i in range(shots):
            positive_few_shot.append(self.get_positive_few_shot_example(feature, eval_prompt, shots=1))
            # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
            positive_few_shot[i] = '\n'.join(positive_few_shot[i])
            negative_few_shot.append(self.get_negative_few_shot_example(feature, eval_prompt, shots=1))
            # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
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


    def createPromptRandom(self,eval_prompt, feature, shots):
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
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot = []
        negative_few_shot = []
        eval_string = f"""
                Me: Answer with Yes or No if this feature:
                    {feature_description}\n
                is present in the following prompt:\n"""
        for i in range(shots):
            positive_few_shot.append(self.get_positive_few_shot_example(feature, eval_prompt, shots=1))
            # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
            positive_few_shot[i] = '\n'.join(positive_few_shot[i])
            negative_few_shot.append(self.get_negative_few_shot_example(feature, eval_prompt, shots=1))
            # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
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


    def createPromptRevised(self,eval_prompt, feature, shots):
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot1 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot1 = '\n'.join(positive_few_shot1)
        negative_few_shot1 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
        # negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
        negative_few_shot1 = '\n'.join(negative_few_shot1)

        positive_few_shot2 = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        # positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot2 = '\n'.join(positive_few_shot2)
        negative_few_shot2 = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
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


    def createPromptGregor(self,eval_prompt, feature, shots):
        feature_description = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]
        # include = self.FEATURES.loc[self.FEATURES['feature_name'] == feature]['include'].iloc[0]
        positive_few_shot = self.get_positive_few_shot_example(feature, eval_prompt, shots=shots)
        positive_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(positive_few_shot)]
        positive_few_shot = '\n'.join(positive_few_shot)
        negative_few_shot = self.get_negative_few_shot_example(feature, eval_prompt, shots=shots)
        negative_few_shot = ['Prompt ' + str(idx + 1) + ': ' + val for idx, val in enumerate(negative_few_shot)]
        negative_few_shot = '\n'.join(negative_few_shot)

        eval_string = f"""Given the following feature:
                {feature_description}\n
                The feature is present in the following prompts:
                {positive_few_shot}\n
                The feature is not present in the following prompts:
                {negative_few_shot}\n
                Tell me whether the feature is present in the prompt given below. Formalize your output as a json object, where the key is the feature description and the associated value is 1 if the feature is present or 0 if not.\n
                Prompt:
                {eval_prompt}"""
        return eval_string, feature_description


    def getPrompt(self):
        if self.promptCreatorid == 0:
            return self. createPromptZero(eval_prompt, feature, shots)
        elif self.promptCreatorid == 1:
            return createPrompt(eval_prompt, feature, shots)
        elif self.promptCreatorid == 2:
            return  createPromptInverted(eval_prompt, feature, shots)
        elif self.promptCreatorid == 3:
            return self.createPromptRevised(eval_prompt, feature, shots)
        elif self.promptCreatorid == 4:
            return  self.createPromptRandom(eval_prompt, feature, shots)
        elif self.promptCreatorid == 5:
            return  self.createPromptRandom2(eval_prompt, feature, shots)
        elif self.promptCreatorid == 6:
            return  self.createPromptRandom3(eval_prompt, feature, shots)