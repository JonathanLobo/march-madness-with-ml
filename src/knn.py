# import numpy as np
# from sklearn import neighbors, datasets
# from sklearn.metrics import accuracy_score
#
# knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# knn.fit(xTrain, yTrain)
# pred = knn.predict(xTest)
#
# print(accuracy_score(yTest, pred))

import data
import csv
import math
import operator
import itertools

def getDistance(item1, item2, length):
	distance = 0
	for x in range(length):
		distance += pow((item1[x] - item2[x]), 2)
	return math.sqrt(distance)

def getWeight(item1, item2, bandwidth, length):
	num = 0 - pow(getDistance(item1, item2, length), 2)
	den = pow(bandwidth, 2)
	weight = math.exp(num/den)
	return weight

def getNeighbors(trainingX, testItem, k, length):
	distances = []
	for x in range(len(trainingX)):
		dist = getDistance(testItem, trainingX[x], length)
		distances.append((trainingX[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getNeighborsGaussian(trainingX, testItem, bandwidth, length):
	weights = []
	for x in range(len(trainingX)):
		weight = getWeight(testItem, trainingX[x], bandwidth, length)
		weights.append(weight)
	return weights

def predictLabel(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def predictLabelGaussian(weights, trainingX):
	classVotes = {}
	for x in range(len(weights)):
		label = trainingX[x][-1]
		if label in classVotes:
			classVotes[label] += weights[x]
		else:
			classVotes[label] = weights[x]
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testingX, predictions):
	correct = 0
	for x in range(len(testingX)):
		if testingX[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testingX))) * 100.0

def normalize(trainingX, testingX):
	for i in range(len(trainingX[0]) - 1):
		summation = 0
		for j in range(len(trainingX)):
			summation = summation + float(trainingX[j][i])
		mean = summation/len(trainingX)

		sumSquares = 0
		for k in range(len(trainingX)):
			sumSquares = sumSquares + pow(float(trainingX[k][i]) - mean, 2)
		meanSumSquares = sumSquares/len(trainingX)
		stDev = math.sqrt(meanSumSquares)

		for m in range(len(trainingX)):
			trainingX[m][i] = (float(trainingX[m][i]) - mean)/stDev

		for n in range(len(testingX)):
			testingX[n][i] = (float(testingX[n][i]) - mean)/stDev

	return trainingX, testingX

if __name__ == "__main__":

	trainingX, trainingY, team_stats = data.get_data()

	tourney_teams, team_id_map = data.get_tourney_teams(2017)
	tourney_teams.sort()

	testingXtemp = []

	matchups = []

	for team1 in tourney_teams:
		for team2 in tourney_teams:
			if team1 < team2:
				game_features = data.get_game_features(team_1, team_2, 0, 2017, team_stats)
				testingXtemp.append(game_features)

				game = [team_1, team_2]
				matchups.append(game)

	testingX = np.array(testingXtemp)

	# mode = 0 is KNN, mode = 1 is Gaussian Weighted Nearest Neighbors
	mode = 0
	k = 10

	for i in range(len(trainingY)):
		trainingX[i].append(trainingY[i])

	trainingX, testingX = normalize(trainingSet, testSet)
	trainingY = []

	predictions=[]

	for x in range(len(testingX)):
		if mode == 1:
			neighbors = getNeighborsGaussian(trainingX, testingX[x], k, len(testingX[x]))
			result = predictLabelGaussian(neighbors, trainingX)
		else:
			neighbors = getNeighbors(trainingX, testingX[x], k, len(testingX[x]))
			result = predictLabel(neighbors)
		predictions.append(result)

	for i in range(0, len(matchups)):
		matchups[i].append(predictions[i])

	results = np.array(matchups)
	np.savetxt("KNN_Predictions_2017.csv", results, delimiter=",", fmt='%s')

	# accuracy = getAccuracy(testingX, predictions)
