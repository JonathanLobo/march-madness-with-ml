import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score

knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
knn.fit(xTrain, yTrain)
pred = knn.predict(xTest)

print(accuracy_score(yTest, pred))
