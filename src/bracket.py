import pandas as pd
import data

def build_bracket():
    teams_df = pd.read_csv("../data2017/FirstRound.csv", header=0)
    tourney_teams, team_id_map = data.get_tourney_teams(2017)

    adaboost_data = pd.read_csv("../predictions/AdaBoost_Predictions_2017.csv", header=0)
    knn_data = pd.read_csv("../predictions/KNN_Predictions_2017.csv", header=0)
    naive_bayes_data = pd.read_csv("../predictions/NaiveBayes_Predictions_2017.csv", header=0)
    neural_net_data = pd.read_csv("../predictions/NeuralNet_Predictions_2017.csv", header=0)
    random_forest_data = pd.read_csv("../predictions/RandomForest_Predictions_2017.csv", header=0)
    regression_data = pd.read_csv("../predictions/Regression_Predictions_2017.csv", header=0)
    svm_data = pd.read_csv("../predictions/SVM_Predictions_2017.csv", header=0)

    # test_list = [adaboost_data, knn_data, naive_bayes_data, neural_net_data, random_forest_data, regression_data, svm_data]
    test_list = [neural_net_data]
    teams = []
    for index, row in teams_df.iterrows():
        teams.append(row["Team1"])
        teams.append(row["Team2"])

    my_index = 0
    tempTeams = []
    for i in range(6):  # 6 rounds of tourney
        print
        print "ROUND ", i+1
        while my_index < len(teams):
            #print my_index
            team1 = teams[my_index]
            team2 = teams[my_index+1]
            #print "t1: ", team1, " t2: ", team2

            for index, row in random_forest_data.iterrows():
                #print my_index
                if (row["Team1"] == team1 and row["Team2"] == team2) or (row["Team1"] == team2 and row["Team2"] == team1):
                    if row["Prediction"] == 0:
                        tempTeams.append(row["Team1"])
                        print team_id_map[row["Team1"]], " over ", team_id_map[row["Team2"]]
                    else:
                        tempTeams.append(row["Team2"])
                        print team_id_map[row["Team2"]], " over ", team_id_map[row["Team1"]]
                    my_index = my_index + 2

        teams = tempTeams
        tempTeams = []
        my_index = 0

if __name__ == "__main__":
    build_bracket()
