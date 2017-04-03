import numpy as np                              # used to support arrays
from sklearn import svm                         # SVM library
from sklearn.metrics import accuracy_score      # SVM accuracy metric

#initialize all kernels
model_rbf = svm.SVC(kernel='rbf')
model_linear = svm.SVC(kernel='linear')
model_poly = svm.SVC(kernel='poly')
model_sigmoid = svm.SVC(kernel='sigmoid')

# train all models on training set
model_rbf.fit(training, np.ravel(trainLabels))
model_linear.fit(training, np.ravel(trainLabels))
model_poly.fit(training, np.ravel(trainLabels))
model_sigmoid.fit(training, np.ravel(trainLabels))

# predict on training set
predict_rbf_train = model_rbf.predict(training)
predict_linear_train = model_linear.predict(training)
predict_poly_train = model_poly.predict(training)
predict_sigmoid_train = model_sigmoid.predict(training)

print(str(accuracy_score(trainLabels, predict_rbf_train)))
print(str(accuracy_score(trainLabels, predict_linear_train)))
print(str(accuracy_score(trainLabels, predict_poly_train)))
print(str(accuracy_score(trainLabels, predict_sigmoid_train)))

# predict on testing set
predict_rbf = model_rbf.predict(testing)
predict_linear = model_linear.predict(testing)
predict_poly = model_poly.predict(testing)
predict_sigmoid = model_sigmoid.predict(testing)

print(str(accuracy_score(testLabels, predict_rbf)))
print(str(accuracy_score(testLabels, predict_linear)))
print(str(accuracy_score(testLabels, predict_poly)))
print(str(accuracy_score(testLabels, predict_sigmoid)))
