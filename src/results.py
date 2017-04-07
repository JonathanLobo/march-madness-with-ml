import data
import numpy as np

if __name__ == "__main__":

    trainingX, trainingY, team_stats = data.get_data()

    tourney_teams, team_id_map = data.get_tourney_teams(2017)
    tourney_teams.sort()

    testingXtemp = []

    matchups = []
    matchups.append(['Team1', 'Team2'])

    for team1 in tourney_teams:
        for team2 in tourney_teams:
            if team1 < team2:
                game_features = data.get_game_features(team_1, team_2, 0, 2017, team_stats)
                testingXtemp.append(game_features)

                game = [team_1, team_2]
                matchups.append(game)

    testingX = np.array(testingXtemp)

    # here is where you train the model using trainingX and trainingY
    # then make output label predictions based on testingX

    # assuming that predictions is an array of the output labels
    # name the output file something else
    for i in range(0, len(matchups)):
        matchups[i].append(predictions[i])

    results = np.array(matchups)
    np.savetxt("ModelName.csv", results, delimiter=",", fmt='%s')
