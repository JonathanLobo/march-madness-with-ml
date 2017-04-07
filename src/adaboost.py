import data
import csv				# used for reading in CSV
import numpy as np		# used for array operations
import math				# used for basic mathematical functions

def getBestStump(stumps, dataLabels, weights):

	bestD = 0
	bestT = 0
	minError = float('inf')

	for d in range(0, D):	# D feature dimensions
		for t in range(0, T):	# T thresholds
			weightedError = 0
			for n in range(0, N):	# N samples
				I = 0
				if (stumps[d][t][n] != dataLabels[n]):
					I = 1
				weightedError = weightedError + weights[n] * I
			if (weightedError < minError):
				bestD = d
				bestT = t
				minError = weightedError
			# print(minError)
	# print(bestD)
	# print(bestT)
	bestStump = stumps[bestD][bestT]

	return bestD, bestT, bestStump, minError

def adaBoost(M, D, T, N, trainingX, thresholds):
	classifierAlphas = np.zeros(M)
	classifierD = np.zeros(M)
	classifierT = np.zeros(M)

	# AdaBoost Loop
	for m in range (0, M):	# M rounds
		# Array is 8 x 10 x 380. In each index D, T we store an array of 380 class predictions
		stumps = np.zeros((D, T, N))
		for d in range(0, D):	# D feature dimensions
			for t in range(0, T):	# T thresholds
				predictions = np.zeros(N)
				for n in range (0, N):	# N training samples
					# print(trainingX[n][d])
					# print(thresholds[d][t])
					if (trainingX[n][d] > thresholds[d][t]):
						predictions[n] = 1
					else:
						predictions[n] = 0
				stumps[d][t] = predictions
		# choose the best stump and calculate error and alpha
		bestD, bestT, bestStump, error = getBestStump(stumps, trainingY, weights)
		errorNorm = error / np.sum(weights)
		alpha = math.log((1 - errorNorm) / errorNorm)

		# update weights using alpha
		for n in range(0, N):	# N samples
			I = 0
			if (bestStump[n] != trainingY[n]):
				I = 1
			weights[n] = weights[n] * math.exp(alpha * I)

		# normalize weights
		sumWeights = np.sum(weights)
		for n in range(0, N):
			weights[n] = weights[n] / sumWeights

		# store the classifier and weights used
		classifierAlphas[m] = alpha
		classifierD[m] = bestD
		classifierT[m] = bestT

	return classifierD, classifierT, classifierAlphas

def predict(dataSet, classD, classT, thresholds, alphas):

	N = len(dataSet)
	M = len(alphas)

	predictions = np.zeros(N)

	for n in range(0, N):	# loop through data set
		total = 0
		for m in range(0, M):	# loop through all weak learners
			# print(n)
			# print(classD[m])
			# print(dataSet[n][int(classD[m])])
			# print(thresholds[int(classD[m])][int(classT[m])])
			# exit()
			if (dataSet[n][int(classD[m])] > thresholds[int(classD[m])][int(classT[m])]):
				ym = 1
			else:
				ym = -1
			total = total + alphas[m] * ym
		if (total > 1):
			predictions[n] = 1
		else:
			predictions[n] = 0

	return predictions

def getAccuracy(predictions, labels):
	N = len(predictions)
	numCorrect = 0

	for n in range(0, N):
		if (predictions[n] == labels[n]):
			numCorrect = numCorrect + 1

	accuracy = numCorrect / N
	return accuracy

if __name__ == '__main__':

	trainingX, trainingY, team_stats = data.get_data()

	print("Generated training set!")

	tourney_teams, team_id_map = data.get_tourney_teams(2017)
	tourney_teams.sort()

	print("Got tourney teams!")

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
	testingY = []

	print("Generated testing set!")

	N = len(trainingX)
	weights = np.ones(N)/N

	M = 30	# number of rounds
	D = len(trainingX[0]) # number of feature dimensions
	T = 10	# number of thresholds per feature dimension

	mins = np.min(trainingX, axis=0)
	maxes = np.max(trainingX, axis=0)
	thresholds = np.zeros((D, T))

	# initialize 10 different threshold values for each dimension
	for d in range(0, D):
		thresholds[d] = np.linspace(mins[d], maxes[d], T, endpoint=False)

	# call Adaptive Boost loop
	classifierD, classifierT, classifierAlphas = adaBoost(M, D, T, N, trainingX, thresholds)

	print("Done fitting the model!")

	# make predictions
	testPredictions = predict(testingX, classifierD, classifierT, thresholds, classifierAlphas)

	print("Finished making predictions!")

	for i in range(0, len(matchups)):
	    matchups[i].append(testPredictions[i])

	results = np.array(matchups)
	np.savetxt("AdaBoost_Predictions_2017.csv", results, delimiter=",", fmt='%s')

	# Print the classifiers and weights used
	print("Classifiers Used:")
	for i in range(0, len(classifierAlphas)):
		print("Feature Dimension: " + str(int(classifierD[i] + 1)) + ", Threshold: " + str(thresholds[int(classifierD[i])][int(classifierT[i])]) + ", Alpha: " + str(classifierAlphas[i]))

	# Calculate and print test accuracy
	# testAccuracy = getAccuracy(testPredictions, testingY)
	# print("Test Set Accuracy: " + str(testAccuracy))
