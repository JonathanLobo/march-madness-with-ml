import data
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
	trainingX, trainingY, team_stats = data.get_data()

	print("Generated training set!")

	tourney_teams, team_id_map = data.get_tourney_teams(2017)
	tourney_teams.sort()

	print("Got tourney teams!")

	testingXtemp = []
	testingYtemp = []

	matchups = []

	for team_1 in tourney_teams:
	    for team_2 in tourney_teams:
	        if team_1 < team_2:
	            game_features = data.get_game_features(team_1, team_2, 0, 2017, team_stats)
	            testingXtemp.append(game_features)

	            game = [team_1, team_2]
	            matchups.append(game)

	testingX = np.array(testingXtemp)
	# testingY = np.array(testingYtemp)

	print("Generated testing set!")

	model_adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(), learning_rate=0.1)
	model_adaboost.fit(trainingX, np.ravel(trainingY))

	print("Done fitting the model!")

	# make predictions
	# testPredictions = model_adaboost.predict(testingX)
	testPredictions = model_adaboost.predict_proba(testingX)

	print("Finished AdaBoost predictions!")

	for i in range(0, len(matchups)):
	    # matchups[i].append(testPredictions[i])
		matchups[i].append(testPredictions[i][0])

	results = np.array(matchups)
	np.savetxt("AdaBoost_Probs_2017.csv", results, delimiter=",", fmt='%s')
