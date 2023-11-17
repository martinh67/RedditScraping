# import statements required
import time
import requests
from datetime import datetime
import pickle
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import unidecode
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import string
from RedditComment import RedditComment

from nltk.stem import WordNetLemmatizer, PorterStemmer

# download wordnet
nltk.download('wordnet')

# list of stopwords
stop_words = set(stopwords.words("english"))

# declare WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# declare stemmer
# removed due to accuracy_score decrease
ps = PorterStemmer()


# get data from the website
def pull_shift_pull(subreddit, start_stamp, end_stamp):

    # url injection
    url = "https://api.pushshift.io/reddit/search/?limit=1000&after={}&before={}&subreddit={}"

    # declare a list class
    list_class = []

    # while the starting date is less than the end date
    while start_stamp < end_stamp:

        # stop bombardment of reddit
        time.sleep(1)

        # update the url with the variables
        update_url = url.format(start_stamp, end_stamp, subreddit)

        # make a json request
        json = requests.get(update_url)

        # pass data to json object
        json_data = json.json()

        # if data is not in json data
        if "data" not in json_data:

            # break from the loop
            break

        # otherwise
        else:

            # set the json_data to the data from the api call
            json_data = json_data['data']

            # print the length of the data returned
            print(len(json_data))

            # if there is no data
            if len(json_data) == 0:

                # print no more data
                print("no more data to harvest")

                # break from the loop
                break

            # try
            try:

                # set the stamp to the last entry of the list
                start_stamp = json_data[-1]['created_utc']

            # handle exceptions
            except:

                # set the start stamp to the end stamp to end loop
                start_stamp = end_stamp

            # use a list to store the data
            list_class = process_json_data(json_data, list_class)

    # return the list class
    return list_class


# method to process json data
def process_json_data(data, list_class):

    # for all of the items in the data
    for item in data:

        # set a new reddit class object for the submission
        reddit_comment = RedditComment()

        # set the body of the object to the body of the item
        reddit_comment.body = item['body']

        # set the time of the object to the time of the item
        reddit_comment.created_utc = item['created_utc']

        # append the object to the list class
        list_class.append(reddit_comment)

    # return the list class
    return list_class


# method to build the dataframes required
def build_dataframe(pickle_class):

    # create list to hold the dates
    dates = []

    # create a list to hold the body text
    text = []

    # for every object in the later class list
    for item in pickle_class:

        # append the dates to a list
        dates.append(item.created_utc)

        # append the body text to a list
        text.append(item.body)

    # create a dataframe
    df = pd.DataFrame({"dates": dates, "body": text})

    # return the dataframe
    return df


# method to create the file
def create_file(filename, subreddit, start_stamp, end_stamp):

    # declare the list class from the pull_shift_pull method
    list_class = pull_shift_pull(subreddit, start_stamp, end_stamp)

    # open the file for writing
    picklefile = open(f"{filename}", "wb")

    # dump the list class and the file into
    pickle.dump(list_class, picklefile)


# method to predict unseen text with the model
def predict_unseen_text(text, transformer, model):

    # format the text
    text = [f"{text}"]

    # transform the text
    transformed_text = transformer.transform(text)

    # put the transformed text into an array
    vectorised_text = transformed_text.toarray()

    # use the vectorised text with the model to make a prediction
    prediction_unseen = model.predict(vectorised_text)

    # print intro message
    print("The text is predicted to be in: ")

    # if the prediction is a 1
    if prediction_unseen == 1:

        # print that the text is more likely to be in daredevil reddit
        print("Daredevil Reddit")

    # otherwise
    else:

        # print that the text is more likely to be in punisher reddit
        print("Punisher Reddit")


# method to get the transformer using the df
def get_transformer(df_mcu):

    # set the transformer with cleaned text
    transformer = CountVectorizer(analyzer = clean_text).fit(df_mcu['body'])

    # return the transformer
    return transformer


# get the x and y train and test data from the dataframe
def get_train_test_data(df_mcu, transformer):

    # transform the dataframe
    process_transformer = transformer.transform(df_mcu['body'])

    # put the transformed data into an array
    X = process_transformer.toarray()

    # declare the variables used to train and test the model
    x_train, x_test, y_train, y_test = train_test_split(X, df_mcu.is_daredevil,
    test_size = 0.25, random_state = 42)

    # return the train and test data
    return x_train, x_test, y_train, y_test


# create the bernoulli model using the training data
def create_bernoulli_model(x_train, y_train):

    # build the bernoulli naive bayes model using the training data
    model = BernoulliNB(alpha = 0.75).fit(x_train, y_train)

    # return the model
    return model


# prepare the dataframe with both daredevil and punisher i.e., the mcu
def prepare_df(df_daredevil, df_punisher):

    # create a new column in the dataframe in order to classify the subreddits
    df_daredevil['is_daredevil'] = 1
    df_punisher['is_daredevil'] = 0

    # declare a new df that is the concatenation of the two sub
    df_mcu = pd.concat([df_daredevil[:500], df_punisher[:500]], join = "outer")

    # return the marvel cinermatic universe dataframe
    return df_mcu

# method to clean the text
def clean_text(text):

    # declare a sentence array
    sentence = []

    # remove punctuation
    term = [character for character in text if character not in string.punctuation]

    # join the terms
    term = ''.join(term)

    # remove non-alphabetic characters and convert terms to lowecase
    term = re.sub('[^a-zA-Z]|(\w+:\/\/\S+)', ' ', term.lower())

    # lemmatise words
    term = lemmatizer.lemmatize(term, pos = 'v')

    # check if word is > 2 and < 20 characters long
    if len(term) > 2 and len(term) < 20:

        # add term to list
        sentence.append(term)

    # form sentence
    sentence = [word for word in term.split() if word not in stop_words]

    # return sentence
    return sentence
