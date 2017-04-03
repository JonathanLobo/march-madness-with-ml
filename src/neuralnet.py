from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fix(xTrain, yTrain)
predictions = mlp.predict(xTest)
