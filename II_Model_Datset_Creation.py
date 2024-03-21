#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:14:08 2024

@author: mfeene
"""

# -------------------------
# II. DATA MANIPULATION
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# -------------------------
# Data Loading 
# -------------------------

games_with_season_stats = pd.read_csv('/Users/mfeene/Desktop/marchmadness_2024/games_with_season_stats_addl.csv')
del games_with_season_stats['Unnamed: 0']

# -------------------------
# Data Manipulation 
# -------------------------
model_data = games_with_season_stats[['Season', 'WTeamID', 'LTeamID']]
model_data = model_data.rename(columns = {'WTeamID': 'Team1', 'LTeamID': 'Team2'})

model_data['WLoc'] = games_with_season_stats['WLoc']
model_data['WinsDiff'] = games_with_season_stats['WTeam_Wins'] - games_with_season_stats['LTeam_Wins']
model_data['LossesDiff'] = games_with_season_stats['WTeam_Losses'] - games_with_season_stats['LTeam_Losses']

model_data['Pom_AvgRankDiff'] = games_with_season_stats['WTeam_Pom_AvgRank'] - games_with_season_stats['LTeam_Pom_AvgRank']
model_data['Pom_BestRankDiff'] = games_with_season_stats['WTeam_Pom_BestRank'] - games_with_season_stats['LTeam_Pom_BestRank']
model_data['Pom_WorstRankDiff'] = games_with_season_stats['WTeam_Pom_WorstRank'] - games_with_season_stats['LTeam_Pom_WorstRank']
model_data['Mas_AvgRankDiff'] = games_with_season_stats['WTeam_Mas_AvgRank'] - games_with_season_stats['LTeam_Mas_AvgRank']
model_data['Mas_BestRankDiff'] = games_with_season_stats['WTeam_Mas_BestRank'] - games_with_season_stats['LTeam_Mas_BestRank']
model_data['Mas_WorstRankDiff'] = games_with_season_stats['WTeam_Mas_WorstRank'] - games_with_season_stats['LTeam_Mas_WorstRank']
model_data['Rpi_AvgRankDiff'] = games_with_season_stats['WTeam_Rpi_AvgRank'] - games_with_season_stats['LTeam_Rpi_AvgRank']
model_data['Rpi_BestRankDiff'] = games_with_season_stats['WTeam_Rpi_BestRank'] - games_with_season_stats['LTeam_Rpi_BestRank']
model_data['Rpi_WorstRankDiff'] = games_with_season_stats['WTeam_Rpi_WorstRank'] - games_with_season_stats['LTeam_Rpi_WorstRank']
model_data['ExpCoachDiff'] = games_with_season_stats['WTeam_Exp_Coach'] - games_with_season_stats['LTeam_Exp_Coach']
model_data['RollingWinDiff'] = games_with_season_stats['WTeam_rolling_win_percentage'] - games_with_season_stats['LTeam_rolling_win_percentage']
model_data['Last10WinPercentageDiff'] = games_with_season_stats['WTeam_season_last10_wins_percentage'] - games_with_season_stats['LTeam_season_last10_wins_percentage']                       
model_data['Last10ConsecWinDiff'] = games_with_season_stats['WTeam_last10_consecutive_wins'] - games_with_season_stats['LTeam_last10_consecutive_wins']

model_data['WinPctDiff'] = games_with_season_stats['WTeam_WinPct'] - games_with_season_stats['LTeam_WinPct']
model_data['FGRatioDiff'] = games_with_season_stats['WTeam_FG_ratio'] - games_with_season_stats['LTeam_FG_ratio']
model_data['3PTDiff'] = games_with_season_stats['WTeam_3PT_ratio'] - games_with_season_stats['LTeam_3PT_ratio']
model_data['ATORatioDiff'] = games_with_season_stats['WTeam_ATO_ratio'] - games_with_season_stats['LTeam_ATO_ratio']
model_data['Outcome'] = 1


# -------------------------
# Data Manipulation 2
# -------------------------
# Randomly sample 50% of the dataframe to simulate the other team winning
# Multiply the diff variables by -1
model_data_to_swap = model_data.sample(frac = 0.5, random_state = 123)

model_data_to_swap['WinsDiff'] = model_data_to_swap['WinsDiff']*-1
model_data_to_swap['LossesDiff'] = model_data_to_swap['LossesDiff']*-1

model_data_to_swap['Pom_AvgRankDiff'] = model_data_to_swap['Pom_AvgRankDiff']*-1
model_data_to_swap['Pom_BestRankDiff'] = model_data_to_swap['Pom_BestRankDiff']*-1
model_data_to_swap['Pom_WorstRankDiff'] = model_data_to_swap['Pom_WorstRankDiff']*-1
model_data_to_swap['Mas_AvgRankDiff'] = model_data_to_swap['Mas_AvgRankDiff']*-1
model_data_to_swap['Mas_BestRankDiff'] = model_data_to_swap['Mas_BestRankDiff']*-1
model_data_to_swap['Mas_WorstRankDiff'] = model_data_to_swap['Mas_WorstRankDiff']*-1
model_data_to_swap['Rpi_AvgRankDiff'] = model_data_to_swap['Rpi_AvgRankDiff']*-1
model_data_to_swap['Rpi_BestRankDiff'] = model_data_to_swap['Rpi_BestRankDiff']*-1
model_data_to_swap['Rpi_WorstRankDiff'] = model_data_to_swap['Rpi_WorstRankDiff']*-1
model_data_to_swap['ExpCoachDiff'] = model_data_to_swap['ExpCoachDiff']*-1
model_data_to_swap['RollingWinDiff'] = model_data_to_swap['RollingWinDiff']*-1
model_data_to_swap['Last10WinPercentageDiff'] = model_data_to_swap['Last10WinPercentageDiff']*-1
model_data_to_swap['Last10ConsecWinDiff'] = model_data_to_swap['Last10ConsecWinDiff']*-1

model_data_to_swap['WinPctDiff'] = model_data_to_swap['WinPctDiff']*-1
model_data_to_swap['FGRatioDiff'] = model_data_to_swap['FGRatioDiff']*-1
model_data_to_swap['3PTDiff'] = model_data_to_swap['3PTDiff']*-1
model_data_to_swap['ATORatioDiff'] = model_data_to_swap['ATORatioDiff']*-1

# Swap the positions of WTeamID and LTeamID
model_data_to_swap = model_data_to_swap[['Season', 'Team2', 'Team1', 'WLoc', 'WinsDiff', 'LossesDiff', 
                                         
                                         'Pom_AvgRankDiff', 'Pom_BestRankDiff', 'Pom_WorstRankDiff',
                                         'Mas_AvgRankDiff', 'Mas_BestRankDiff', 'Mas_WorstRankDiff',
                                         'Rpi_AvgRankDiff', 'Rpi_BestRankDiff', 'Rpi_WorstRankDiff',
                                         'ExpCoachDiff', 'RollingWinDiff', 'Last10WinPercentageDiff', 'Last10ConsecWinDiff',
                            
                                         'WinPctDiff', 'FGRatioDiff', '3PTDiff', 'ATORatioDiff', 
                                         'Outcome']]

# These will have their outcome variables switched to 0
model_data_to_swap['Outcome'] = model_data_to_swap['Outcome'].replace([1], [0])

# Rest of the dataframe, leave as is
model_data_orig = model_data.loc[~model_data.index.isin(model_data_to_swap.index)]


# -------------------------
# Final Modeling Dataset
# -------------------------
## Put the two dataframes back together
final_model_data = pd.concat([model_data_orig, model_data_to_swap], axis = 0)
final_model_data.to_csv('/Users/mfeene/Desktop/marchmadness_2024/final_model_data_addl.csv')