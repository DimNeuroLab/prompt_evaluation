import os


def get_api_key():
    """
    Get API key from a text file. Search for directory in file path, and stop if directory named
    "prompt_evaluation" is found. Then, read API key from text file named key.txt located in
    this directory.

    Returns:
        str: the key
    """
    # Get the absolute path of current working directory
    absolute_path = os.path.abspath(os.getcwd())

    # Split the path by separator and stop the loop when 'prompt_evaluation' directory is found
    directory = [f for f in absolute_path.split(os.sep) if f != "prompt_evaluation"]
    directory.append('prompt_evaluation')

    # Join the directory list to form absolute path of required directory
    absolute_path = os.sep.join(directory)

    # Read the API key from 'key.txt' in the above directory
    with open(os.path.join(absolute_path, "key.txt"), "r") as key_file:
        key = key_file.read()

    return key
