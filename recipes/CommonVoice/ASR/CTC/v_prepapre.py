import os
import csv
import re
import logging
import torchaudio
import unicodedata
import pandas as pd 


logger = logging.getLogger(__name__)


def prepare_common_voice(
    data_folder = None, 
    train_csv = None,
    valid_csv= None,
    test_csv= None,
    save_folder=".",
    skip_prep = False,
    accented_letters = True,
    language = "fr"
):

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_csv is None:
        train_csv = data_folder + "/train.csv"
    else:
        train_csv = train_csv

    if valid_csv is None:
        valid_csv = data_folder + "/valid.csv"
    else:
        valid_csv = valid_csv

    if test_csv is None:
        test_csv = data_folder + "/test.tsv"
    else:
        test_csv = test_csv


    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_valid = save_folder + "/valid.csv"
    save_csv_test = save_folder + "/test.csv"

    if skip(save_csv_train, save_csv_valid, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_valid)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return


    if train_csv is not None:

        create_csv(
            train_csv,
            save_csv_train,
            data_folder,
            accented_letters,
            language,
        )

    if valid_csv is not None:

        create_csv(
            valid_csv, save_csv_valid, data_folder, accented_letters, language
        )

    # Creating csv file for test data
    if test_tsv_file is not None:

        create_csv(
            test_csv,
            save_csv_test,
            data_folder,
            accented_letters,
            language,
        )

def create_csv(
    orig_csv_file, csv_file, data_folder, accented_letters=True,language="fr"):


    if not os.path.isfile(orig_csv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_csv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    loaded_csv = pd.read_csv(orig_csv_file).dropna()

    # We load and skip the header
    # loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "wrd"]]

    # Start processing lines
    total_duration = 0.0
    for index, row in loaded_csv.iterrows():
        words = row['wrd']
        # print(row)
        # print("wwwwww",words,not words,row['ID']) 
        if language in ["en", "fr", "it", "rw"]:
            words = re.sub(
                "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
            ).upper()

        if language == "fr":
            # Replace J'y D'hui etc by J_ D_hui
            words = words.replace("'", " ")
            words = words.replace("’", " ")
        
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        # if len(words.split(" ")) < 3:
        #     continue
                # Composition of the csv_line
        mp3_path = os.path.abspath(data_folder)+"/"+row["ID"]+".wav"

        csv_line = [str(row["ID"]), str(row["duration"]), mp3_path, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)
        # Writing the csv lines
   
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)


def skip(save_csv_train, save_csv_valid, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_valid)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip



def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)