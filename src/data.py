import pandas as pd
import random
import numpy as np

team_stats = {}
stat_fields = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr',
    'ast', 'to', 'stl', 'blk', 'pf', 'o_score', 'o_fgm', 'o_fga','o_fgm3', 'o_fga3',
    'o_ftm', 'o_fta', 'o_or', 'o_dr', 'o_ast', 'o_to', 'o_stl','o_blk','o_pf']

tourneyYear = '2017'

# return csv as a pandas dataframe
def format_as_df(csv_file):
    df = pd.read_csv(csv_file)
    return df

def get_dataframes():
    season_compact_results = d.format_as_df('../data' + tourneyYear + '/RegularSeasonCompactResults.csv')
    season_detailed_results = d.format_as_df('../data' + tourneyYear + '/RegularSeasonDetailedResults.csv')
    teams = d.format_as_df('../data' + tourneyYear + '/Teams.csv')
    seasons = d.format_as_df('../data' + tourneyYear + '/Seasons.csv')
    tourney_compact_results = d.format_as_df('../data' + tourneyYear + '/TourneyCompactResults.csv')
    tourney_detailed_results = d.format_as_df('../data' + tourneyYear + '/TourneyDetailedResults.csv')
    seeds = d.format_as_df('../data' + tourneyYear + '/TourneySeeds.csv')
    slots = d.format_as_df('../data' + tourneyYear + '/TourneySlots.csv')

# get temporary team stats for a current point in the season while generating test cases
def get_stat_temp(season, team, field):
    try:
        stat = team_stats[season][team][field]
        return sum(stat) / float(len(stat))
    except:
        return 0

# get final team stats for a season, passing in an array of team stats, to use for prediction
def get_stat_final(season, team, field, team_stats):
    try:
        stat = team_stats[season][team][field]
        return sum(stat) / float(len(stat))
    except:
        return 0

# update team stats based on most recent game
def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        # We only want to keep track of n most recent games, so get rid of the oldest if necessary
        if len(team_stats[season][team][key]) >= 15:
            team_stats[season][team][key].pop()

        team_stats[season][team][key].append(value)

# use outside of this file when we want to get game features so that we can predict using any model
def get_game_features(team_1, team_2, loc, season, all_stats):
    # both teams are "away" since it is a tourney game
    if (loc == 1):
        # first team is home, second is away
        features = [0, 1]
    elif (loc == -1):
        # first team is away, second is home
        features = [1, 0]
    else:
        # both teams are "away" (neutral)
        features = [1, 1]

    # Team 1
    for stat in stat_fields:
        features.append(get_stat_final(season, team_1, stat, all_stats))

    # Team 2
    for stat in stat_fields:
        features.append(get_stat_final(season, team_2, stat, all_stats))

    return features

# get the ids of all teams in the tourney and a map of team ids to their actual name
def get_tourney_teams(year):
    seeds = pd.read_csv('../data' + tourneyYear + '/TourneySeeds.csv')
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == year:
            tourney_teams.append(row['Team'])

    team_id_map = get_team_dict()

    return tourney_teams, team_id_map

# get a map of team ids to team names
def get_team_dict():
    teams = pd.read_csv('../data' + tourneyYear + '/Teams.csv')
    team_map = {}
    for index, row in teams.iterrows():
        team_map[row['Team_Id']] = row['Team_Name']
    return team_map

# build test cases for all seasons
def build_season_data(data):
    X = []
    y = []

    for index, row in data.iterrows():

        skip = 0
        firstLoc = 0
        secondLoc = 0
        randNum = random.random()

        # home = 0, away and neutral = 1
        if (randNum > 0.5):
            if (row['Wloc'] == 'H'):
                # first team is home, second is away
                firstLoc = 0
                secondLoc = 1
            elif (row['Wloc'] == 'A'):
                # first team is away, second is home
                firstLoc = 1
                secondLoc = 0
            else:
                # both teams are "away" (neutral)
                firstLoc = 1
                secondLoc = 1
        else:
            if (row['Wloc'] == 'H'):
                # first team is away, second is home
                firstLoc = 1
                secondLoc = 0
            elif (row['Wloc'] == 'A'):
                # first team is home, second is away
                firstLoc = 0
                secondLoc = 1
            else:
                # both teams are "away" (neutral)
                firstLoc = 1
                secondLoc = 1

        loc = [firstLoc, secondLoc]

        team_1_features = []
        team_2_features = []

        # get all other team statistics
        for field in stat_fields:
            team_1_stat = get_stat_temp(row['Season'], row['Wteam'], field)
            team_2_stat = get_stat_temp(row['Season'], row['Lteam'], field)

            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        # if skip = 0, it is the first game of the season so we have no prior statistics (everything is 0)
        # We label as '0' if team1 won, label as '1' if team2 won
        if skip == 0:
            if randNum > 0.5:
                X.append(loc + team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(loc + team_2_features + team_1_features)
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
            'pf': row['Wpf'],
            'o_score': row['Lscore'],
            'o_fgm': row['Lfgm'],
            'o_fga': row['Lfga'],
            'o_fgm3': row['Lfgm3'],
            'o_fga3': row['Lfga3'],
            'o_ftm': row['Lftm'],
            'o_fta': row['Lfta'],
            'o_or': row['Lor'],
            'o_dr': row['Ldr'],
            'o_ast': row['Last'],
            'o_to': row['Lto'],
            'o_stl': row['Lstl'],
            'o_blk': row['Lblk'],
            'o_pf': row['Lpf']
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
            'pf': row['Lpf'],
            'o_score': row['Wscore'],
            'o_fgm': row['Wfgm'],
            'o_fga': row['Wfga'],
            'o_fgm3': row['Wfgm3'],
            'o_fga3': row['Wfga3'],
            'o_ftm': row['Wftm'],
            'o_fta': row['Wfta'],
            'o_or': row['Wor'],
            'o_dr': row['Wdr'],
            'o_ast': row['Wast'],
            'o_to': row['Wto'],
            'o_stl': row['Wstl'],
            'o_blk': row['Wblk'],
            'o_pf': row['Wpf']
        }
        update_stats(row['Season'], row['Wteam'], stat_1_fields)
        update_stats(row['Season'], row['Lteam'], stat_2_fields)

    trainX = np.array(X)
    trainY = np.array(y)

    return trainX, trainY

def get_data():
    year = int(tourneyYear)

    for i in range(1985, year+1):
        team_stats[i] = {}

    season_detailed_results = format_as_df('../data' + tourneyYear + '/RegularSeasonDetailedResults.csv')
    tourney_detailed_results = format_as_df('../data' + tourneyYear + '/TourneyDetailedResults.csv')
    frames = [season_detailed_results, tourney_detailed_results]
    data = pd.concat(frames)

    X, Y = build_season_data(data)

    return X, Y, team_stats

if __name__ == "__main__":
    trainingX, trainingY, team_stats = get_data()
