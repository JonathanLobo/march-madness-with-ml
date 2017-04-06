import pandas as pd
import random

team_stats = {}
stat_fields = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']

def format_as_df(csv_file):
    df = pd.read_csv(csv_file)
    return df

def get_dataframes():
    season_compact_results = d.format_as_df('../data2016/RegularSeasonCompactResults.csv')
    season_detailed_results = d.format_as_df('../data2016/RegularSeasonDetailedResults.csv')
    teams = d.format_as_df('../data2016/Teams.csv')
    seasons = d.format_as_df('../data2016/Seasons.csv')
    tourney_compact_results = d.format_as_df('../data2016/TourneyCompactResults.csv')
    tourney_detailed_results = d.format_as_df('../data2016/TourneyDetailedResults.csv')
    seeds = d.format_as_df('../data2016/TourneySeeds.csv')
    slots = d.format_as_df('../data2016/TourneySlots.csv')

def get_stat_temp(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0

def get_stat_final(season, team, field, team_stats):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0

# use outside of this file when we want to get game features so that we can predict using any model
def get_game_features(team_1, team_2, season, team_stats):
    features = []

    # Team 1
    for stat in stat_fields:
        features.append(get_stat_final(season, team_1, stat, team_stats))

    # Team 2
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat, team_stats))

    return features

def get_tourney_teams(year):
    seeds = pd.read_csv('../data2016/TourneySeeds.csv')
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == year:
            tourney_teams.append(row['Team'])

    team_id_map = get_team_dict()

    return tourney_teams, team_id_map

def get_team_dict():
    team_ids = pd.read_csv('../data2016//Teams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['Team_Id']] = row['Team_Name']
    return team_id_map

def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        team_stats[season][team][key].append(value)

def build_season_data(data):
    X = []
    y = []

    for index, row in data.iterrows():

        skip = 0

        team_1_features = []
        team_2_features = []

        for field in stat_fields:
            team_1_stat = get_stat_temp(row['Season'], row['Wteam'], field)
            team_2_stat = get_stat_temp(row['Season'], row['Lteam'], field)

            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        # if skip = 0, it is the first game of the season so we have no prior statistics (everything is 0)
        if skip == 0:
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)

        # Update teams' overall stats so that they can later be averaged and used to make predictions
        stat_1_fields = {
            'score': row['Wscore'],
            'fgm': row['Wfgm'],
            'fga': row['Wfga'],
            'fgm3': row['Wfgm3'],
            'fga3': row['Wfga3'],
            'ftm': row['Wftm'],
            'fta': row['Wfta'],
            'or': row['Wor'],
            'dr': row['Wdr'],
            'ast': row['Wast'],
            'to': row['Wto'],
            'stl': row['Wstl'],
            'blk': row['Wblk'],
            'pf': row['Wpf']
        }
        stat_2_fields = {
            'score': row['Lscore'],
            'fgm': row['Lfgm'],
            'fga': row['Lfga'],
            'fgm3': row['Lfgm3'],
            'fga3': row['Lfga3'],
            'ftm': row['Lftm'],
            'fta': row['Lfta'],
            'or': row['Lor'],
            'dr': row['Ldr'],
            'ast': row['Last'],
            'to': row['Lto'],
            'stl': row['Lstl'],
            'blk': row['Lblk'],
            'pf': row['Lpf']
        }
        update_stats(row['Season'], row['Wteam'], stat_1_fields)
        update_stats(row['Season'], row['Lteam'], stat_2_fields)

    return X, y

def get_data():
    year = 2016

    for i in range(1985, year+1):
        team_stats[i] = {}

    season_detailed_results = format_as_df('../data2016/RegularSeasonDetailedResults.csv')
    tourney_detailed_results = format_as_df('../data2016/TourneyDetailedResults.csv')
    frames = [season_detailed_results, tourney_detailed_results]
    data = pd.concat(frames)

    X, Y = build_season_data(data)

    return X, Y, team_stats

if __name__ == "__main__":
    trainingX, trainingY, team_stats = get_data()
