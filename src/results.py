import pandas as pd

def compute_accuracies():
    test_Y = pd.read_csv("../data2017/TourneyMatchups2017.csv", header=0)

    adaboost_data = pd.read_csv("../predictions/AdaBoost_Predictions_2017.csv", header=0)
    knn_data = pd.read_csv("../predictions/KNN_Predictions_2017.csv", header=0)
    naive_bayes_data = pd.read_csv("../predictions/NaiveBayes_Predictions_2017.csv", header=0)
    neural_net_data = pd.read_csv("../predictions/NeuralNet_Predictions_2017.csv", header=0)
    random_forest_data = pd.read_csv("../predictions/RandomForest_Predictions_2017.csv", header=0)
    regression_data = pd.read_csv("../predictions/Regression_Predictions_2017.csv", header=0)
    svm_data = pd.read_csv("../predictions/SVM_Predictions_2017.csv", header=0)

    test_list = [adaboost_data, knn_data, naive_bayes_data, neural_net_data, random_forest_data, regression_data, svm_data]

    count = 0
    for df in test_list:
        for index, row in df.iterrows():
            for index, test_row in test_Y.iterrows():
                if row['Team1'] == test_row['Team1'] and row['Team2'] == test_row['Team2']:  # make sure teams match
                    if row['Prediction'] == test_row['Prediction']:
                        # print "Team 1: ", row['Team1']
                        # print "Team 2: ", row['Team2']
                        # print "Winner: ", row['Prediction']
                        count = count + 1
        print ("Accuracy: " + str(count / 63.0 * 100))
        count = 0

if __name__ == "__main__":
    compute_accuracies()
