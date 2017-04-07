import data
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train():
    print "Building Data..."
    trainingX, trainingY, team_stats = data.get_data()

    tourney_teams, team_id_map = data.get_tourney_teams(2017)
    tourney_teams.sort()

    testingXtemp = []

    matchups = []

    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                game_features = data.get_game_features(team_1, team_2, 0, 2017, team_stats)
                testingXtemp.append(game_features)

                game = [team_1, team_2]
                matchups.append(game)

    testingX = np.array(testingXtemp)

    print "Fitting model..."
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(trainingX, trainingY)

    return model, testingX, matchups

def predict(model, test_data, matchups):
    print "Generating predictions..."
    predictions = model.predict(test_data)

    # assuming that predictions is an array of the output labels
    # name the output file something else
    for i in range(0, len(matchups)):
        matchups[i].append(predictions[i])

    results = np.array(matchups)
    np.savetxt("neuralnet_predictions_2017.csv", results, delimiter=",", fmt='%s')

if __name__ == "__main__":
    model, test_data, matchups = train()
    predict(model, test_data, matchups)

