# import numpy as np
# from sklearn import neighbors, datasets
# from sklearn.metrics import accuracy_score
#
# knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# knn.fit(xTrain, yTrain)
# pred = knn.predict(xTest)
#
# print(accuracy_score(yTest, pred))

import csv
import math
import operator
import itertools

def readDataset(filename, dataSet=[] , dataLabels=[]):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(760):
			tempList=[]
			for y in range(8):
				tempList.append(float(dataset[x][y]))
			dataSet.append(tempList)
			dataLabels.append(str(dataset[x][8]))

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

def getNeighbors(trainingSet, testItem, k, length):
	distances = []
	for x in range(len(trainingSet)):
		dist = getDistance(testItem, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getNeighborsGaussian(trainingSet, testItem, bandwidth, length):
	weights = []
	for x in range(len(trainingSet)):
		weight = getWeight(testItem, trainingSet[x], bandwidth, length)
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

def predictLabelGaussian(weights, trainingSet):
	classVotes = {}
	for x in range(len(weights)):
		label = trainingSet[x][-1]
		if label in classVotes:
			classVotes[label] += weights[x]
		else:
			classVotes[label] = weights[x]
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def normalize(trainingSet, testSet):
	for i in range(len(trainingSet[0]) - 1):
		summation = 0
		for j in range(len(trainingSet)):
			summation = summation + float(trainingSet[j][i])
		mean = summation/len(trainingSet)

		sumSquares = 0
		for k in range(len(trainingSet)):
			sumSquares = sumSquares + pow(float(trainingSet[k][i]) - mean, 2)
		meanSumSquares = sumSquares/len(trainingSet)
		stDev = math.sqrt(meanSumSquares)

		for m in range(len(trainingSet)):
			trainingSet[m][i] = (float(trainingSet[m][i]) - mean)/stDev

		for n in range(len(testSet)):
			testSet[n][i] = (float(testSet[n][i]) - mean)/stDev

	return trainingSet, testSet

def runKnn(k, x_train, y_train, x_test):
	for i in range(len(y_train)):
		x_train[i].append(dataLabels[i])

	y_test=[]

	for x in range(len(x_test)):
		neighbors = getNeighbors(x_train, x_test[x], k, len(testSet[x])-1)
		result = predictLabel(neighbors)
		y_test.append(result)
		print('predicted=' + str(result))

	return y_test

def runKnnGaussian(bandwidth, x_train, y_train, x_test):
	for i in range(len(y_train)):
		x_train[i].append(dataLabels[i])

	y_test=[]

	for x in range(len(x_test)):
		neighbors = getNeighborsGaussian(x_train, x_test[x], bandwidth, len(testSet[x])-1)
		result = predictLabelGaussian(neighbors, trainingSet)
		y_test.append(result)
		print('predicted=' + str(result))

	return y_test

# mode = 0 is KNN, mode = 1 is Gaussian weighted nearest neighbors
def testKnn(k, dataSet, dataLabels, mode):

	for i in range(len(dataLabels)):
		dataSet[i].append(dataLabels[i])

	index = 76
	fold1 = dataSet[0:index]
	fold2 = dataSet[index:index+76]
	index = index + 76
	fold3 = dataSet[index:index+76]
	index = index + 76
	fold4 = dataSet[index:index+76]
	index = index + 76
	fold5 = dataSet[index:index+76]
	index = index + 76
	fold6 = dataSet[index:index+76]
	index = index + 76
	fold7 = dataSet[index:index+76]
	index = index + 76
	fold8 = dataSet[index:index+76]
	index = index + 76
	fold9 = dataSet[index:index+76]
	index = index + 76
	fold10 = dataSet[index:index+76]

	listOfFolds=[]
	listOfFolds.append(fold1)
	listOfFolds.append(fold2)
	listOfFolds.append(fold3)
	listOfFolds.append(fold4)
	listOfFolds.append(fold5)
	listOfFolds.append(fold6)
	listOfFolds.append(fold7)
	listOfFolds.append(fold8)
	listOfFolds.append(fold9)
	listOfFolds.append(fold10)

	sumAcc = 0

	for i in range(10):
		trainingSet=[]
		testSet=[]
		for j in range(10):
			if(j == i):
				testSet = listOfFolds[j]
			else:
				trainingSet = trainingSet + listOfFolds[j]

		predictions=[]

		trainingSet, testSet = normalize(trainingSet, testSet)

		for x in range(len(testSet)):
			if mode == 1:
				neighbors = getNeighborsGaussian(trainingSet, testSet[x], k, len(testSet[x])-1)
				result = predictLabelGaussian(neighbors, trainingSet)
			else:
				neighbors = getNeighbors(trainingSet, testSet[x], k, len(testSet[x])-1)
				result = predictLabel(neighbors)
			predictions.append(result)
			# print('predicted=' + str(result) + ', actual=' + str(testSet[x][-1]))

		accuracy = getAccuracy(testSet, predictions)
		sumAcc = sumAcc + accuracy
		print('Run ' + str(i+1) + ' Accuracy: ' + str(round(accuracy, 2)) + '%')

	print('Average Accuracy: ' + str(round(sumAcc/10, 2)) + '%')
	return predictions

dataSet=[]
dataLabels=[]

readDataset('pima-indians-diabetes.data', dataSet, dataLabels)

testKnn(1, dataSet, dataLabels, 1)
