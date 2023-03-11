import os
import csv
import pandas
import numpy as np
import pytesseract
from PIL import Image
from preprocess_tools import *
from correction_tools import *

"""
TODOS:
1. Loop through directories
2. Get all ".jpg" and run script
3. Save all files in a TXT and CSV directory in the main directory using same structure as the one of looped directories
"""

def clean_text_column(image):
    df = pytesseract.image_to_data(image, output_type='data.frame')

    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=["text"])
    # Confidence value:
    df = df[df['conf'] > 25]

    column_as_reference = df['block_num'].ffill()
    output = df.groupby(column_as_reference, sort=False).first()
    output['height'] = round(
        df['height'].dropna().groupby(column_as_reference).mean(), 2)
    output['width'] = round(df['width'].dropna().groupby(
        column_as_reference).mean(), 2)
    output['left'] = round(df['left'].dropna().groupby(
        column_as_reference).mean(), 2)
    output['top'] = round(df['top'].dropna().groupby(
        column_as_reference).mean(), 2)
    output['conf'] = round(df['conf'].dropna().groupby(
        column_as_reference).mean(), 2)
    output['text'] = df['text'].dropna().groupby(
        column_as_reference).agg(' '.join)
    output = output[output['text'].apply(lambda x: len(x.split(' ')) > 3)]
    output.drop(columns=output.columns[:6], axis=1, inplace=True)
    output = output.reset_index(drop=True)
    return output


def df_to_text_file(input, column_name, output_filename):
    # Extract the "text" column from the DataFrame
    text = input[str(column_name)].tolist()
    # Write the elements in the "text" column to a .txt file
    with open(str(output_filename) + ".txt", "w") as f:
        for row in text:
            f.write(row + "\n")


def image_to_txt_and_csv(input, output_filename):
    image = preprocess(input)
    df = clean_text_column(image)

    for i in range(len(df['text'])):
        df.iloc[int(i), 5] = correct_line(unite_sign(
            str(df.iloc[int(i), 5])), "combined_big_text.txt")

    df.to_csv(str(output_filename) + ".csv", index=False)
    df_to_text_file(df, "text", str(output_filename))


##### TESTING #####
image_to_txt_and_csv(get_fullpath(os.getcwd(), "test.jpg"), "testing")
print("allgood")
