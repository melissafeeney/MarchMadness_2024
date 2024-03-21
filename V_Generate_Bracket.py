#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:14:08 2024

@author: mfeene
"""

# -------------------------
# IV. GENERATE 2024 BRACKETS
# -------------------------

#! pip install binarytree==6.2.0
#! pip install bracketeer==0.2.0
#! pip install setuptools_scm==6.0.1

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)
import pandas as pd
import pickle
from bracketeer import build_bracket


# Read in data
submissions = pd.read_csv('/content/SampleSubmission2024.csv')
data = pd.read_csv('/content/team_stats_ratios_2024_addl.csv')
del data['Unnamed: 0']

# Model trained on historical data from 2003, apply to the 2023 matchups
# split game identifier by specified delimiter
matchups = pd.DataFrame()
matchups[['Season','Team1ID', 'Team2ID']] = submissions.ID.str.split("_", expand = True)
matchups['Season'] = pd.to_numeric(matchups['Season'])
matchups['Team1ID'] = pd.to_numeric(matchups['Team1ID'])
matchups['Team2ID'] = pd.to_numeric(matchups['Team2ID'])

# Bring in the team stats for 2023
team1_metrics_added = pd.merge(matchups, data, how = 'left', left_on = ['Season', 'Team1ID'], right_on = ['Season', 'TeamID'])
team1_metrics_added.rename(columns = {'Wins': 'Team1_Wins',
                                          'Losses': 'Team1_Losses',
                                          'Pom_AvgRank': 'Team1_Pom_AvgRank',
                                          'Pom_BestRank': 'Team1_Pom_BestRank',
                                          'Pom_WorstRank': 'Team1_Pom_WorstRank',

                                          'Mas_AvgRank': 'Team1_Mas_AvgRank',
                                          'Mas_BestRank': 'Team1_Mas_BestRank',
                                          'Mas_WorstRank': 'Team1_Mas_WorstRank',

                                          'Rpi_AvgRank': 'Team1_Rpi_AvgRank',
                                          'Rpi_BestRank': 'Team1_Rpi_BestRank',
                                          'Rpi_WorstRank': 'Team1_Rpi_WorstRank',

                                          'exp_coach': 'Team1_Exp_Coach',

                                          'rolling_win_percentage': 'Team1_rolling_win_percentage',
                                          'season_last10_wins_percentage': 'Team1_season_last10_wins_percentage',
                                          'last10_consecutive_wins': 'Team1_last10_consecutive_wins',

                                          'WinPct': 'Team1_WinPct',
                                          'FG_ratio' : 'Team1_FG_ratio',
                                          '3PT_ratio': 'Team1_3PT_ratio',
                                          'ATO_ratio': 'Team1_ATO_ratio'}, inplace = True)

team2_metrics_added = pd.merge(team1_metrics_added, data, how = 'left', left_on = ['Season', 'Team2ID'], right_on = ['Season', 'TeamID'])
team2_metrics_added.rename(columns = {'Wins': 'Team2_Wins',
                                          'Losses': 'Team2_Losses',

                                          'Pom_AvgRank': 'Team2_Pom_AvgRank',
                                          'Pom_BestRank': 'Team2_Pom_BestRank',
                                          'Pom_WorstRank': 'Team2_Pom_WorstRank',

                                          'Mas_AvgRank': 'Team2_Mas_AvgRank',
                                          'Mas_BestRank': 'Team2_Mas_BestRank',
                                          'Mas_WorstRank': 'Team2_Mas_WorstRank',

                                          'Rpi_AvgRank': 'Team2_Rpi_AvgRank',
                                          'Rpi_BestRank': 'Team2_Rpi_BestRank',
                                          'Rpi_WorstRank': 'Team2_Rpi_WorstRank',

                                          'exp_coach': 'Team2_Exp_Coach',

                                          'rolling_win_percentage': 'Team2_rolling_win_percentage',
                                          'season_last10_wins_percentage': 'Team2_season_last10_wins_percentage',
                                          'last10_consecutive_wins': 'Team2_last10_consecutive_wins',

                                          'WinPct': 'Team2_WinPct',
                                          'FG_ratio' : 'Team2_FG_ratio',
                                          '3PT_ratio': 'Team2_3PT_ratio',
                                          'ATO_ratio': 'Team2_ATO_ratio'}, inplace = True)
# Add this in
team2_metrics_added['WLoc'] = 0


# Get data into this format for the 2023
model_2024_data = pd.DataFrame()
model_2024_data = team2_metrics_added[['Season', 'Team1ID', 'Team2ID']]

model_2024_data['WLoc'] = team2_metrics_added['WLoc']
model_2024_data['WinsDiff'] = team2_metrics_added['Team1_Wins'] - team2_metrics_added['Team2_Wins']
model_2024_data['LossesDiff'] = team2_metrics_added['Team1_Losses'] - team2_metrics_added['Team2_Losses']
model_2024_data['Pom_AvgRankDiff'] = team2_metrics_added['Team1_Pom_AvgRank'] - team2_metrics_added['Team2_Pom_AvgRank']
model_2024_data['Pom_BestRankDiff'] = team2_metrics_added['Team1_Pom_BestRank'] - team2_metrics_added['Team2_Pom_BestRank']
model_2024_data['Pom_WorstRankDiff'] = team2_metrics_added['Team1_Pom_WorstRank'] - team2_metrics_added['Team2_Pom_WorstRank']
model_2024_data['Mas_AvgRankDiff'] = team2_metrics_added['Team1_Mas_AvgRank'] - team2_metrics_added['Team2_Mas_AvgRank']
model_2024_data['Mas_BestRankDiff'] = team2_metrics_added['Team1_Mas_BestRank'] - team2_metrics_added['Team2_Mas_BestRank']
model_2024_data['Mas_WorstRankDiff'] = team2_metrics_added['Team1_Mas_WorstRank'] - team2_metrics_added['Team2_Mas_WorstRank']
model_2024_data['Rpi_AvgRankDiff'] = team2_metrics_added['Team1_Rpi_AvgRank'] - team2_metrics_added['Team2_Rpi_AvgRank']
model_2024_data['Rpi_BestRankDiff'] = team2_metrics_added['Team1_Rpi_BestRank'] - team2_metrics_added['Team2_Rpi_BestRank']
model_2024_data['Rpi_WorstRankDiff'] = team2_metrics_added['Team1_Rpi_WorstRank'] - team2_metrics_added['Team2_Rpi_WorstRank']
model_2024_data['ExpCoachDiff'] = team2_metrics_added['Team1_Exp_Coach'] - team2_metrics_added['Team2_Exp_Coach']
model_2024_data['RollingWinDiff'] = team2_metrics_added['Team1_rolling_win_percentage'] - team2_metrics_added['Team2_rolling_win_percentage']
model_2024_data['Last10WinPercentageDiff'] = team2_metrics_added['Team1_season_last10_wins_percentage'] - team2_metrics_added['Team2_season_last10_wins_percentage']
model_2024_data['Last10ConsecWinDiff'] = team2_metrics_added['Team1_last10_consecutive_wins'] - team2_metrics_added['Team2_last10_consecutive_wins']
model_2024_data['WinPctDiff'] = team2_metrics_added['Team1_WinPct'] - team2_metrics_added['Team2_WinPct']
model_2024_data['FGRatioDiff'] = team2_metrics_added['Team1_FG_ratio'] - team2_metrics_added['Team2_FG_ratio']
model_2024_data['3PTDiff'] = team2_metrics_added['Team1_3PT_ratio'] - team2_metrics_added['Team2_3PT_ratio']
model_2024_data['ATORatioDiff'] = team2_metrics_added['Team1_ATO_ratio'] - team2_metrics_added['Team2_ATO_ratio']


# -------------------------
# Generating Predictions
# -------------------------

# Read in data
X = model_2024_data.iloc[:, 3:].values

# Load scaler
scalerfile = 'scaler.save'
sc = pickle.load(open(scalerfile, 'rb'))

# Apply scaler to data
X = sc.transform(X)

# Predict probabilities of team 1 winning each matchup
predictions = predict_stacked_model(stacked_model, X) # this function is designed inn the model creation files
predictions = predictions.flatten()


# -------------------------
# 2024 Predictions
# -------------------------
# Read in sample predictions
template = pd.read_csv('/content/SampleSubmission2024.csv')
del template['Unnamed: 0']
spreadsheet = template.iloc[:, :].values

# Add the prediction to each prediction index
for i in range(0, len(predictions)):
     spreadsheet[i][1] = round(predictions[i], 5)

# Create dataframe to match sample submissions spreadsheet
results = pd.DataFrame(data = spreadsheet, columns=['ID', 'Pred'])

# Save new submissions spreadsheet as csv
results.to_csv('Submission2024.csv', sep = ',', encoding = 'utf-8', index = False)


b = build_bracket(
    outputPath='non_attention_MNCAA2024.png',
    teamsPath='/content/MTeams.csv',
    seedsPath='/content/MNCAATourneySeeds.csv',
    submissionPath='/content/Submission2024.csv',
    slotsPath='/content/MNCAATourneySlots.csv',
    year=2024
)