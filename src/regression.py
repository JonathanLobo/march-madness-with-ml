import data as d
import numpy as np
import pandas as pd

from sklearn import cross_validation, linear_model

def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])

def main():
    year = 2017
    X, y = d.build_season_data(year)
    model = linear_model.LogisticRegression()
    results = []

    print(cross_validation.cross_val_score(
        model, numpy.array(X), numpy.array(y), cv=10, scoring='accuracy', n_jobs=-1
    ).mean())

    model.fit(X, y)

    seeds = pd.DataFrame.from_csv('../data2016/TourneySeeds.csv', header=0)

    teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == year:
            teams.append(row['Team'])

    teams.sort()
    for team_1 in teams:
        for team_2 in teams:
            if team_1 < team_2:
                prediction = predict_winner(team_1, team_2, model, year, stat_fields)
                label = str(year) + '_' + str(team_1) + '_' + str(team_2)
                results.append([label, prediction[0][0]])
                
    print results

if __name__ == "__main__":
    main()
