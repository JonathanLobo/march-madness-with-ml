import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

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
                        # print ("Team 1: " + str(row['Team1']))
                        # print ("Team 2: " + str(row['Team2']))
                        # print ("Winner: " + str(row['Prediction']))
                        count = count + 1
        print ("Accuracy: " + str(count / 63.0 * 100))
        count = 0

def compute_log_loss():
    tourney_games = pd.read_csv("../data2017/TourneyMatchups2017.csv", header=0)

    adaboost_data = pd.read_csv("../predictions/AdaBoost_Probs_2017.csv", header=0)
    knn_data = pd.read_csv("../predictions/KNN_Probs_2017.csv", header=0)
    naive_bayes_data = pd.read_csv("../predictions/NaiveBayes_Probs_2017.csv", header=0)
    neural_net_data = pd.read_csv("../predictions/NeuralNet_Probs_2017.csv", header=0)
    random_forest_data = pd.read_csv("../predictions/RandomForest_Probs_2017.csv", header=0)
    regression_data = pd.read_csv("../predictions/Regression_Probs_2017.csv", header=0)
    svm_data = pd.read_csv("../predictions/SVM_Probs_2017.csv", header=0)

    test_list = [adaboost_data, knn_data, naive_bayes_data, neural_net_data, random_forest_data, regression_data, svm_data]

    for df in test_list:
        y_true_list = []
        y_pred_list = []
        for index, row in df.iterrows():
            for index, test_row in tourney_games.iterrows():
                if row['Team1'] == test_row['Team1'] and row['Team2'] == test_row['Team2']:  # make sure teams match

                    team2Prob = 1 - row['Team1Prob']

                    # print ("Team 1: " +  str(row['Team1']))
                    # print ("Team 2: " + str(row['Team2']))
                    # print(team2Prob)

                    y_true_list.append(test_row['Prediction'])
                    y_pred_list.append(team2Prob)

        y_true = np.asarray(y_true_list)
        y_pred = np.asarray(y_pred_list)
        l_loss = log_loss(y_true, y_pred)
        print ("Log Loss: " + str(l_loss))

if __name__ == "__main__":
    compute_accuracies()
    print()
    compute_log_loss()
