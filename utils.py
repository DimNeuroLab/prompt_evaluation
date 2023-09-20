import os


def get_api_key():
    absolute_path = os.path.abspath(os.getcwd())
    directory = []
    for f in absolute_path.split('/'):
        directory.append(f)
        if f == 'prompt_evaluation':
            break
    absolute_path = '/'.join(directory)
    with open(absolute_path + '/key.txt', 'r') as key_file:
        key = key_file.read()
    return key
