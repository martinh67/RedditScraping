# import statements required
import time
import pickle
from methods import *
from RedditComment import RedditComment

# start the timer
start = time.time()


# main method
def main():

    '''
    # uncomment this code to build the files with many comments
    daredevil = create_file("daredevil.pkl", "Daredevil", 1554073200, 1651338156)
    punisher = create_file("punisher.pkl", "thepunisher", 1554073200, 1651338156)
    '''

    # open the daredevil file for reading
    daredevil_file = open("daredevil.pkl", "rb")
    daredevil_class = pickle.load(daredevil_file)

    # open the daredevil file for reading
    punisher_file = open("punisher.pkl", "rb")
    punisher_class = pickle.load(punisher_file)

    # build the dataframes
    df_daredevil = build_dataframe(daredevil_class)
    df_punisher = build_dataframe(punisher_class)

    # declare a new df that is the concatenation of the two subreddits
    df_mcu = prepare_df(df_daredevil, df_punisher)

    # get the transformer from the dataframe
    transformer = get_transformer(df_mcu)

    # get the test and training data required
    x_train, x_test, y_train, y_test = get_train_test_data(df_mcu, transformer)

    # create the model
    model = create_bernoulli_model(x_train, y_train)

    # create the prediction
    prediction = model.predict(x_test)

    # print the evaluation of the model prediciton
    print()
    print("BernoulliNB Model Evaluation:")
    print(f"Accuracy: {accuracy_score(prediction, y_test)}")
    print()

    # give an example of unseen text
    unseen_text = "Why was Agent Madani in Season 2"

    print()
    print(f"Unseen text is: {unseen_text}")

    # print the model's prediction on unseen text
    print()
    predict_unseen_text(unseen_text, transformer, model)
    print()



# magic method to run the main function
if __name__ == "__main__":

    # run main
    main()


# print the time of the program
print("\n" + 40*"#")
print(time.time() - start)
print(40*"#" + "\n")
