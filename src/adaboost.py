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

def adaBoost(M, D, T, N, trainingSet, thresholds):
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
					# print(trainingSet[n][d])
					# print(thresholds[d][t])
					if (trainingSet[n][d] > thresholds[d][t]):
						predictions[n] = 1
					else:
						predictions[n] = 0
				stumps[d][t] = predictions
		# choose the best stump and calculate error and alpha
		bestD, bestT, bestStump, error = getBestStump(stumps, trainingLabels, weights)
		errorNorm = error / np.sum(weights)
		alpha = math.log((1 - errorNorm) / errorNorm)

		# update weights using alpha
		for n in range(0, N):	# N samples
			I = 0
			if (bestStump[n] != dataLabels[n]):
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
	# print decimals, not in scientific notation
	np.set_printoptions(suppress=True)

	dataSet=[]
	dataLabels=[]

	trainingSet = dataSet[0:int(len(dataSet)/2)]
	trainingLabels = dataLabels[0:int(len(dataSet)/2)]
	testSet = dataSet[int(len(dataSet)/2):len(dataSet)]
	testLabels = dataLabels[int(len(dataSet)/2):len(dataSet)]

	N = len(trainingSet)
	weights = np.ones(N)/N

	M = 30	# number of rounds
	D = 8	# number of feature dimensions
	T = 10	# number of thresholds per feature dimension

	# ranges = np.ptp(trainingSet, axis=0)
	# means = np.mean(trainingSet, axis=0)
	# stdevs = np.std(trainingSet, axis=0)
	mins = np.min(trainingSet, axis=0)
	maxes = np.max(trainingSet, axis=0)
	thresholds = np.zeros((D, T))

	# initialize 10 different threshold values for each dimension
	for d in range(0, D):
		thresholds[d] = np.linspace(mins[d], maxes[d], T, endpoint=False)

	# call Adaptive Boost loop
	classifierD, classifierT, classifierAlphas = adaBoost(M, D, T, N, trainingSet, thresholds)

	# print(classifierAlphas)
	# print(classifierD)
	# print(classifierT)
	# print(thresholds)

	# make predictions
	trainingPredictions = predict(trainingSet, classifierD, classifierT, thresholds, classifierAlphas)
	testPredictions = predict(testSet, classifierD, classifierT, thresholds, classifierAlphas)
	# print(testPredictions)

	# Print the classifiers and weights used
	print("Classifiers Used:")
	for i in range(0, len(classifierAlphas)):
		print("Feature Dimension: " + str(int(classifierD[i] + 1)) + ", Threshold: " + str(thresholds[int(classifierD[i])][int(classifierT[i])]) + ", Alpha: " + str(classifierAlphas[i]))

	# Calculate and print accuracy
	trainingAccuracy = getAccuracy(trainingPredictions, trainingLabels)
	testAccuracy = getAccuracy(testPredictions, testLabels)
	print("\nTraining Set Accuracy: " + str(trainingAccuracy))
	print("Test Set Accuracy: " + str(testAccuracy))
