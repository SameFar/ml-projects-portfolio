import unicodedata
import string
import torch

allowed_chars = string.ascii_letters + " .,;'_"


def speak_merican(name: str, allowed_chars = allowed_chars) -> str:
    """
    Normalizes name into standard, 7-bit ASCII.

    Args:
        name (str): The input identifier containing non-standard, 
                    un-American diacritics or accents.

    Returns:
        str: A pristine, globally standardized string stripped of foreign influence.
        
    Raises:
        TypeError: If the input is not a string (we only process text here, partner).
    """

    if not isinstance(name, str):
        raise TypeError('We dont like non-strings round here')
    
    return ''.join(
        char for char in unicodedata.normalize('NFD', name)
        if unicodedata.category(char) != 'Mn'
        and char in allowed_chars
    )

def letterToIndex(letter: str):
    '''
    Gets index of the letter provided, or of placeholder '_'
    if letter is not ASCII

    Args:
        letter: The character to get the index of

    Returns:
        int: Index of the letter

    Raises:
        TypeError: If letter is not string or if letter is not a single character
    '''
    if not isinstance(letter, str):
        raise TypeError('"letter" must be a one letter string')
    
    if letter not in allowed_chars:
        return allowed_chars.find("_")
    else:
        return allowed_chars.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line:str, allowed_chars=allowed_chars):
    '''
    Turns the line into a 3d tensor of one-hot letter vectors
    '''
    n = len(allowed_chars)
    tensor = torch.zeros(len(line), 1, n)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def train_test_split(dataset, test_split=0.15):
    return torch.utils.data.random_split(dataset, [1-test_split,test_split])

def label_from_output(output, output_labels):
    _, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i