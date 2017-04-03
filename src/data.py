import pandas as pd

def format_as_df(csv_file):
    df = pd.DataFrame.from_csv(csv_file, header=0)

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

def get_stat(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0

def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        if len(team_stats[season][team][key]) >= 9:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)

def build_season_data(year):
    team_stats = {}
    X = []
    y = []
    stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
    for i in range(1985, year+1):
        team_stats[i] = {}
    season_detailed_results = format_as_df('../data2016/RegularSeasonDetailedResults.csv')
    tourney_detailed_results = format_as_df('../data2016/TourneyDetailedResults.csv')
    frames = [season_detailed_results, tourney_detailed_results]
    data = pd.concat(frames)

    for index, row in data.iterrows():
        skip = 0

        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['Wteam'], field)
            team_2_stat = get_stat(row['Season'], row['Lteam'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)

        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['Wfta'] != 0 and row['Lfta'] != 0:
            stat_1_fields = {
                'score': row['Wscore'],
                'fgp': row['Wfgm'] / row['Wfga'] * 100,
                'fga': row['Wfga'],
                'fga3': row['Wfga3'],
                '3pp': row['Wfgm3'] / row['Wfga3'] * 100,
                'ftp': row['Wftm'] / row['Wfta'] * 100,
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
                'fgp': row['Lfgm'] / row['Lfga'] * 100,
                'fga': row['Lfga'],
                'fga3': row['Lfga3'],
                '3pp': row['Lfgm3'] / row['Lfga3'] * 100,
                'ftp': row['Lftm'] / row['Lfta'] * 100,
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
