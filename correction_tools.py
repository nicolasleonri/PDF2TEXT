import re
import numpy as np
from spellchecker import SpellChecker
from string import ascii_letters, printable, punctuation
from collections import Counter

"""
TODOS:
1. Test different parameters
2. Add corrections based on literature
3. Automatically determine best parameters to be used
4. Work around lines with special characters
"""


def find_special_char(word):
    """
    This function checks if a word contains any special characters.

    Parameters:
    word (str): The word to be checked.

    Returns:
    bool: True if the word contains a special character, False otherwise.
    """
    regexp = re.compile('[^0-9a-zA-Z-À-ÿ]+', re.UNICODE)
    return bool(regexp.search(word))


def correct_line(line, dict):
    """
    Correct the spelling mistakes in a string `line`.

    The function first splits the string into words, then iterates over the words.
    If the word contains non-letters, the original line is returned.
    Otherwise, the word is corrected using a spell checker. If the word is
    capitalized and not an acronym, the corrected word is capitalized as well.
    If the word is uppercase, the corrected word is returned in uppercase.
    The corrected word replaces the original word in the line.

    Parameters:
    ----------
    line (str): The input string to be corrected.
    dictionary (txt): Corpus used to determine best words

    Returns:
    -------
    line (str): The corrected string with spelling mistakes corrected.
    """

    spell = SpellChecker(language='es')
    spell.word_frequency.load_text_file(str(dict))
    special_characters = "!@#$%^&*()-+?_=,<>/"

    for idx, val in enumerate(line.split()):
        # Skip the word if it contains non-letter characters
        if find_special_char(val) == True:
            continue
        # Check if the word was not found at all in the corpus
        # Confidence level 0.0
        if spell.word_usage_frequency(str(val)) == 0.0:
            word = spell.correction(val)
            # Capitalize the corrected word if the original word is capitalized (not an acronym)
            if str(val)[0].isupper() and not str(val).isupper() and not word == None:
                word = word.capitalize()
            # Uppercase the corrected word if the original word is uppercase
            elif str(val).isupper() and not word == None:
                word = word.upper()
            # If the correction is None, call the Viterbi segmentation
            if word == None:
                test = call_viterbi_segment(
                    str(val), "combined_big_text.txt", 0.0)
                if test != None:
                    line = line.replace(str(val), str(test))
                    continue
                else:
                    continue
            # Replace the original word with the corrected word
            line = line.replace(str(val), str(word))
        else:
            continue
    return line


def unite_sign(line):
    """
    Unites two words split by a hyphen if the first word ends with an alphabetic character and the second word starts with a numeric character.

    Args:
        line (str): The input line to be processed.

    Returns:
        str: The processed line where two words separated by a hyphen are joined if they meet the specified conditions.

    Example:
        >>> unite_sign("This is an example-5 test.")
        'This is an example-5 test.'
        >>> unite_sign("This is a test-case")
        'This is a test-case'
    """
    # Initialize an empty list to store the processed words
    processed_words = []

    jump = False

    for idx, val in enumerate(line.split()):

        if jump == True:
            jump = False
            continue

        if "-" not in val:
            # If the word does not contain a negative sign, add it to the processed_words list
            processed_words.append(val)
            continue

        elif val[int(str(val).find("-")) - 1].isalpha():
            try:
                # Check if the character following the negative sign is numeric
                if val[int(str(val).find("-")) + 2].isnumeric():
                    processed_words.append(val)
                    continue
            except:
                pass

            val = val.replace("-", "")
            # print(val)
            try:
                # Get the next word from the list
                to_add = line.split()[idx + 1]
                # print(to_add)
            except:
                # Add the current word to the processed_words list and continue if an exception occurs
                processed_words.append(val)
                continue

            # Concatenate the current word and the next word
            new_word = val + to_add
            # Add the new word to the processed_words list
            processed_words.append(new_word)
            # line = line.replace(str(line.split()[idx+1]), "")
            jump = True
            continue
        else:
            # If the negative sign is not within a word, add the word to the processed_words list
            processed_words.append(val)
            continue

    # Join the processed_words list into a single string
    processed_words = " ".join(processed_words)
    return processed_words


def call_viterbi_segment(line, dict, conf):
    """
    Segments a given text into words using the Viterbi algorithm and returns the segmented text.

    Args:
    line (str): The text to be segmented.
    dictionary (dict): A dictionary of words and their frequencies.
    conf (float): The confidence level to be used as reference.

    Returns:
    str: The segmented text.
    float: The confidence level of the segmentation.
    """
    def viterbi_segment(text):
        """
        Uses the Viterbi algorithm to segment the given text into words.

        Args:
        text (str): The text to be segmented.
        word_probs (dict): A dictionary of word probabilities.
        max_word_length (int): The maximum length of a word in the dictionary.

        Returns:
        list: The segmented words.
        float: The confidence level of the segmentation.
        """
        probs, segmentation_points = [1.0], [0]
        for i in range(1, len(text) + 1):
            prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                            for j in range(max(0, i - max_word_length), i))
            probs.append(prob_k)
            segmentation_points.append(k)
        words = []
        i = len(text)
        while 0 < i:
            words.append(text[segmentation_points[i]:i])
            i = segmentation_points[i]
        words.reverse()

        # confidence level
        if probs[-1] == float(conf):
            return None

        return words, probs[-1]

    def word_prob(word):
        return dictionary[word] / total

    def words(text):
        return re.findall('[a-z]+', text.lower())

    dictionary = Counter(
        words(open('combined_big_text.txt', encoding="utf-8").read()))
    max_word_length = max(map(len, dictionary))
    sum = 0
    for item in dictionary.values():
        sum = sum + item
    total = float(sum)
    output = viterbi_segment(line)

    try:
        output = " ".join(output[0])
    except:
        return None

    return output
