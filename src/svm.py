import data
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

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

    #initialize all kernels
    # model_rbf = svm.SVC(kernel='rbf', probability=True)
    model_rbf = svm.SVC(kernel='rbf', probability=True)
    # model_linear = svm.SVC(kernel='linear', probability=True)
    # model_poly = svm.SVC(kernel='poly', probability=True)
    # model_sigmoid = svm.SVC(kernel='sigmoid', probability=True)

    # train all models on training set
    model_rbf.fit(trainingX, np.ravel(trainingY))
    # model_linear.fit(trainingX, np.ravel(trainingY))
    # model_poly.fit(trainingX, np.ravel(trainingY))
    # model_sigmoid.fit(trainingX, np.ravel(trainingY))

    print("Done fitting SVM!")

    # predict on testing set
    # predict_rbf = model_rbf.predict(testingX)
    predict_rbf = model_rbf.predict_proba(testingX)
    # predict_linear = model_linear.predict(testingX)
    # predict_poly = model_poly.predict(testingX)
    # predict_sigmoid = model_sigmoid.predict(testingX)

    print("Finished SVM predictions!")

    for i in range(0, len(matchups)):
        # matchups[i].append(predict_rbf[i])
        matchups[i].append(predict_rbf[i][0])

    results = np.array(matchups)
    np.savetxt("SVM_Probs_2017.csv", results, delimiter=",", fmt='%s')

    # print(str(accuracy_score(testingY, predict_rbf)))
    # print(str(accuracy_score(testingY, predict_linear)))
    # print(str(accuracy_score(testingY, predict_poly)))
    # print(str(accuracy_score(testingY, predict_sigmoid)))
