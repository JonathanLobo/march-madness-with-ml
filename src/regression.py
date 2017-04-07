import data
import numpy as np

from sklearn import linear_model

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

    # here is where you train the model using trainingX and trainingY
    # then make output label predictions based on testingX

    print "Fitting model..."
    model = linear_model.LogisticRegression(C=1e5)
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
    np.savetxt("regression_predictions_2017.csv", results, delimiter=",", fmt='%s')

if __name__ == "__main__":
    model, test_data, matchups = train()
    predict(model, test_data, matchups)
