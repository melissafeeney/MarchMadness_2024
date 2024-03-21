#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:14:08 2024

@author: mfeene
"""

# -------------------------
# I. DATA CLEANING
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data Loading
# -------------------------

# Read in data
path = '/Users/mfeene/Desktop/marchmadness_2024/kaggle_data_31824/'

## Read in Data
#massey_ord = pd.read_csv(path + 'MMasseyOrdinals.csv') # this file will be updated 
massey_ord = pd.read_csv(path + 'MMasseyOrdinals_thruSeason2024_day128.csv') 
regseason = pd.read_csv(path + 'MRegularSeasonDetailedResults.csv')
postseason = pd.read_csv(path + 'MNCAATourneyDetailedResults.csv')
coaches = pd.read_csv(path + 'MTeamCoaches.csv')
seasons = pd.read_csv(path + 'MSeasons.csv', parse_dates = ['DayZero'])
teams = pd.read_csv(path + 'MTeams.csv')


# Only consider 2003 and on- just remove 2024 for here
regseason = regseason[(regseason['Season'] >= 2003)]
postseason = postseason[(postseason['Season'] >= 2003)]
frames = [regseason, postseason]
games = pd.concat(frames)


# -------------------------
#  1. Season Level Team Coaches
# -------------------------
###### if there was more than one coach per season, this will get messed up
# count number of coaches per team per season
# count the number of consecutive seasons that coach of team X coached them- for ex 1102
# Initialize a dictionary to store consecutive years for each coach within each team
team_coaches = coaches[['Season', 'TeamID', 'CoachName']]

years_per_team = {}

# Function to update consecutive years for a coach within each team
def update_years(row):
    team = row['TeamID']
    coach = row['CoachName']
    if team not in years_per_team:
        years_per_team[team] = {}
    if coach in years_per_team[team]:
        years_per_team[team][coach] += 1
    else:
        years_per_team[team][coach] = 1
    return years_per_team[team][coach]

# Apply the function to calculate consecutive years for each row within each team
team_coaches['years_per_team'] = team_coaches.apply(update_years, axis=1)

# Check if there were any teams with more than one coach per season- need to figure out a solve for this- maybe a 0/1 flag to see if a team had a coach who had been there for more than __ years?
team_coaches.groupby(['Season', 'TeamID'])['CoachName'].count().reset_index()

# If a team had a coach who was there for 7+ seasons, give them a 1
exp_coaches = team_coaches[team_coaches['years_per_team'] >= 7]
exp_coaches = exp_coaches[['Season', 'TeamID']]
exp_coaches = exp_coaches.drop_duplicates() # remove dupes; team 1112 had 2 coaches with more than 7 years exp in 2001?
exp_coaches['exp_coach'] = 1



# -------------------------
#  2. Season Level Team Descriptive Stats- Avg, Best and Worst Pom Rankings per Season per Team
# -------------------------
# A. KENPOM
# -------------------------
# Consider Pomeroy Rankings- note that this only goes through the regseason
pom_rank = massey_ord[(massey_ord['SystemName'] == 'POM')]

# For each team in each year, record the avg ranking, best rank, worst rank
pom_avg_rank = pom_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).mean().reset_index()
pom_best_rank = pom_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).min().reset_index()
pom_worst_rank = pom_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).max().reset_index()

pom_avg_rank.columns = ['Season', 'TeamID', 'Pom_AvgRank']
pom_best_rank.columns = ['Season', 'TeamID', 'Pom_BestRank']
pom_worst_rank.columns = ['Season', 'TeamID', 'Pom_WorstRank']

pom_pre_team_ranks = pd.merge(pom_avg_rank, pom_best_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID'])
pom_team_ranks = pd.merge(pom_pre_team_ranks, pom_worst_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID']) ## contains each team's avg, best and worst rank per season


# -------------------------
#  B. MASSEY 
# -------------------------
# Consider Massey Rankings
mas_rank = massey_ord[(massey_ord['SystemName'] == 'MAS')]

# For each team in each year, record the avg ranking, best rank, worst rank
mas_avg_rank = mas_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).mean().reset_index()
mas_best_rank = mas_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).min().reset_index()
mas_worst_rank = mas_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).max().reset_index()

mas_avg_rank.columns = ['Season', 'TeamID', 'Mas_AvgRank']
mas_best_rank.columns = ['Season', 'TeamID', 'Mas_BestRank']
mas_worst_rank.columns = ['Season', 'TeamID', 'Mas_WorstRank']

mas_pre_team_ranks = pd.merge(mas_avg_rank, mas_best_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID'])
mas_team_ranks = pd.merge(mas_pre_team_ranks, mas_worst_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID']) ## contains each team's avg, best and worst rank per season


# -------------------------
#  C. RPI
# -------------------------
# Consider RPI Rankings
rpi_rank = massey_ord[(massey_ord['SystemName'] == 'RPI')]

# For each team in each year, record the avg ranking, best rank, worst rank
rpi_avg_rank = rpi_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).mean().reset_index()
rpi_best_rank = rpi_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).min().reset_index()
rpi_worst_rank = rpi_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).max().reset_index()

rpi_avg_rank.columns = ['Season', 'TeamID', 'Rpi_AvgRank']
rpi_best_rank.columns = ['Season', 'TeamID', 'Rpi_BestRank']
rpi_worst_rank.columns = ['Season', 'TeamID', 'Rpi_WorstRank']

rpi_pre_team_ranks = pd.merge(rpi_avg_rank, rpi_best_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID'])
rpi_team_ranks = pd.merge(rpi_pre_team_ranks, rpi_worst_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID']) ## contains each team's avg, best and worst rank per season



# -------------------------
# 3. Avg Win Streak- At each game in the season, how many of its last 10 games did the winning team win?
# this will need to be aggregated by season level
#### do not include the tournament games here - DayNum <= 132
# -------------------------
# get every team's W/L path each season
team_wins = games[['Season', 'DayNum', 'WTeamID']].groupby(['Season', 'WTeamID', 'DayNum']).size().reset_index()
team_wins.columns = ['Season', 'TeamID', 'DayNum', 'outcome']
team_wins = team_wins[team_wins['DayNum'] <= 132] # reg season
team_wins['outcome'] = team_wins['outcome'].replace(1, 'win')


team_losses = games[['Season', 'DayNum', 'LTeamID']].groupby(['Season', 'LTeamID', 'DayNum']).size().reset_index()
team_losses.columns = ['Season', 'TeamID', 'DayNum', 'outcome']
team_losses = team_losses[team_losses['DayNum'] <= 132] # reg season
team_losses['outcome'] = team_losses['outcome'].replace(1, 'loss')

###
schedule = pd.concat([team_wins, team_losses])
schedule = schedule.sort_values(by=['Season','TeamID','DayNum'])

schedule['outcome'] = schedule['outcome'].map({'win': 1, 'loss': 0})
#del schedule['DayNum']
schedule = schedule.reset_index(drop=True)

rolling_wins = schedule.groupby(['Season', 'TeamID'])['outcome'].rolling(10, min_periods=1).mean().reset_index()
del rolling_wins['level_2']
rolling_wins.columns = ['Season', 'TeamID', 'rolling_win_percentage']

final_schedule = pd.concat([schedule, rolling_wins], axis = 1)
final_schedule.columns = ['Season', 'TeamID', 'DayNum','outcome', 'Season_delete', 'TeamID_delete', 'rolling_win_percentage']
final_schedule = final_schedule[['Season', 'TeamID','DayNum', 'outcome', 'rolling_win_percentage']]

avg_season_rolling_win_percentage = final_schedule.groupby(['Season', 'TeamID'])['rolling_win_percentage'].mean().reset_index()


# -------------------------
# 4. Last 10 Games Win Streak
# tournament is from day 133-154 every season, so take the last 10 before tournament starts
# -------------------------
final_schedule = final_schedule.sort_values(by=['Season', 'TeamID', 'DayNum'])
season_last10 = final_schedule[['Season','TeamID','DayNum','outcome']].groupby(['Season','TeamID']).tail(10)
season_last10_wins = season_last10.groupby(['Season', 'TeamID'])['outcome'].mean().reset_index()
season_last10_wins.columns = ['Season', 'TeamID','season_last10_wins_percentage']


# -------------------------
# 5. Win/Hot Streak- In the final 10 games of the reg season, how many did a team consecutively win?
# -------------------------
season_last10 = final_schedule[['Season','TeamID','DayNum','outcome']].groupby(['Season','TeamID']).tail(10)

# Initialize a dictionary to store consecutive 1s for each team in each season
consecutive_wins = {}

# Iterate through each row of the DataFrame
for index, row in season_last10.iterrows():
    season = row['Season']
    team_id = row['TeamID']
    outcome = row['outcome']
    
    # Check if the team and season combination exists in the dictionary
    if (season, team_id) not in consecutive_wins:
        consecutive_wins[(season, team_id)] = 0
    
    # If outcome is 1, increment consecutive_ones count; otherwise, reset consecutive_ones count
    if outcome == 1:
        consecutive_wins[(season, team_id)] += 1
    else:
        consecutive_wins[(season, team_id)] = 0

# Convert the dictionary to a DataFrame
season_last10_streak = pd.DataFrame.from_dict(consecutive_wins, orient='index', columns=['last10_consecutive_wins']).reset_index()
season_last10_streak[['Season', 'TeamID']] = pd.DataFrame(season_last10_streak['index'].tolist(), index=season_last10_streak.index)
season_last10_streak.drop(columns=['index'], inplace=True)
season_last10_streak = season_last10_streak[['Season', 'TeamID', 'last10_consecutive_wins']]


# -------------------------
#  6. Season Wins and Losses, Win percentage, per Season per Team
# -------------------------
## Wins per season per team
wins_per_season = games[['Season', 'WTeamID']].groupby(['Season', 'WTeamID']).size().reset_index()
wins_per_season.columns = ['Season', 'TeamID', 'Wins']

## Losses per season per team
losses_per_season = games[['Season', 'LTeamID']].groupby(['Season', 'LTeamID']).size().reset_index()
losses_per_season.columns = ['Season', 'TeamID', 'Losses']

# Put all together
wins_losses = pd.merge(wins_per_season, losses_per_season, how = 'outer', on =['Season', 'TeamID'])
wins_losses  = wins_losses.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)

# Calculate wins percentage for each team each season
wins_losses['WinPct'] = wins_losses['Wins'] / (wins_losses['Wins'] + wins_losses['Losses'])



# -------------------------
#  Put Together Season level Team Descriptive Stats- Start with the wins and losses of each team of each season
# -------------------------
# Add Rankings from the 3 Systems
team_stats = pd.merge(wins_losses, pom_team_ranks, how = 'outer', on =['Season', 'TeamID'])
team_stats = pd.merge(team_stats, mas_team_ranks, how = 'outer', on =['Season', 'TeamID'])
team_stats = pd.merge(team_stats, rpi_team_ranks, how = 'outer', on =['Season', 'TeamID'])
team_stats = team_stats.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)

# Add Experienced Coaching data
team_stats = pd.merge(team_stats, exp_coaches, how = 'left',on =['Season', 'TeamID'])
team_stats = team_stats.fillna(0) # fill season/teams without experienced coaches with a 0 in that column

# Calculate wins percentage for each team each season
team_stats['WinPct'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])

# Add Avg Season 10 Game Rolling Win Percentage
team_stats = pd.merge(team_stats, avg_season_rolling_win_percentage, how = 'outer', on =['Season', 'TeamID'])

# Add Last 10 Games Win Percentage
team_stats = pd.merge(team_stats, season_last10_wins, how = 'outer', on =['Season', 'TeamID'])

# Add Win Streak- how many of last 10 games did a team consecutively win?
team_stats = pd.merge(team_stats, season_last10_streak, how = 'outer', on =['Season', 'TeamID'])

team_stats = team_stats.fillna(0)


# -------------------------
#  7. More Team Descriptive Stats per Season per Team- FG ratio, 3PT ratio, ATO ratio
# -------------------------
# Reference the games file
# instances where the team won
winner_metrics = games[['Season', 'WTeamID', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WAst', 'WTO' ]].groupby(['Season', 'WTeamID']).sum().reset_index()
winner_metrics.rename(columns = {'WTeamID': 'TeamID'}, inplace = True)

# instances where the team lost
loser_metrics = games[['Season', 'LTeamID', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LAst', 'LTO']].groupby(['Season', 'LTeamID']).sum().reset_index()
loser_metrics.rename(columns = {'LTeamID': 'TeamID'}, inplace = True)

# Join to get total metrics per season per team
season_ratios = pd.merge(winner_metrics, loser_metrics, how = 'outer', on =['Season', 'TeamID'])
season_ratios = season_ratios.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)
season_ratios['FG_ratio'] = (season_ratios['WFGM'] + season_ratios['LFGM'])/(season_ratios['WFGA'] + season_ratios['LFGA'])
season_ratios['3PT_ratio'] = (season_ratios['WFGM3'] + season_ratios['LFGM3'])/(season_ratios['WFGA3'] + season_ratios['LFGA3'])
season_ratios['ATO_ratio'] = (season_ratios['WAst'] + season_ratios['LAst'])/(season_ratios['WTO'] + season_ratios['LTO'])
season_ratios = season_ratios[['Season', 'TeamID', 'FG_ratio', '3PT_ratio', 'ATO_ratio']]

# Add these ratios to the team_stats file
team_stats_ratios = pd.merge(team_stats, season_ratios, how = 'outer', on =['Season', 'TeamID'] )
team_stats_ratios = team_stats_ratios.fillna(0.0)
team_stats_ratios.to_csv('team_stats_ratios_addl.csv')

# Team Stats for 2024 only
team_stats_ratios_2024 = team_stats_ratios[(team_stats_ratios['Season'] == 2024)]
team_stats_ratios_2024.to_csv('/Users/mfeene/Desktop/marchmadness_2024/team_stats_ratios_2024_addl.csv')
#team_stats_ratios[team_stats_ratios.isna().any(axis=1)]


# -------------------------
#  Game Descriptive Stats- Avg Ranking for winning and losing teams, winner's location, Plus Wins and Losses, Important Ratios, Per Season
# -------------------------
# Winning Team seasonal metrics- Add to Game List
games_with_season_stats = pd.merge(games[['Season', 'WTeamID', 'LTeamID', 'WLoc']], team_stats_ratios, how = 'left', left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
del games_with_season_stats['TeamID']

games_with_season_stats.rename(columns = {'Wins': 'WTeam_Wins',
                                          'Losses': 'WTeam_Losses',
                                          'Pom_AvgRank': 'WTeam_Pom_AvgRank',
                                          'Pom_BestRank': 'WTeam_Pom_BestRank',
                                          'Pom_WorstRank': 'WTeam_Pom_WorstRank',
                                          
                                          'Mas_AvgRank': 'WTeam_Mas_AvgRank',
                                          'Mas_BestRank': 'WTeam_Mas_BestRank',
                                          'Mas_WorstRank': 'WTeam_Mas_WorstRank',
                                          
                                          'Rpi_AvgRank': 'WTeam_Rpi_AvgRank',
                                          'Rpi_BestRank': 'WTeam_Rpi_BestRank',
                                          'Rpi_WorstRank': 'WTeam_Rpi_WorstRank',
                                          
                                          'exp_coach': 'WTeam_Exp_Coach',
                                          
                                          'rolling_win_percentage': 'WTeam_rolling_win_percentage',
                                          'season_last10_wins_percentage': 'WTeam_season_last10_wins_percentage',
                                          'last10_consecutive_wins': 'WTeam_last10_consecutive_wins',
                                          
                                          'WinPct': 'WTeam_WinPct',
                                          'FG_ratio' : 'WTeam_FG_ratio',
                                          '3PT_ratio': 'WTeam_3PT_ratio',
                                          'ATO_ratio': 'WTeam_ATO_ratio'}, inplace = True)

# Losing Team seasonal metrics- Add to Game List
games_with_season_stats = pd.merge(games_with_season_stats, team_stats_ratios, how = 'left', left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
del games_with_season_stats['TeamID']

games_with_season_stats.rename(columns = {'Wins': 'LTeam_Wins',
                                          'Losses': 'LTeam_Losses',
                                          
                                          'Pom_AvgRank': 'LTeam_Pom_AvgRank',
                                          'Pom_BestRank': 'LTeam_Pom_BestRank',
                                          'Pom_WorstRank': 'LTeam_Pom_WorstRank',
                                          
                                          'Mas_AvgRank': 'LTeam_Mas_AvgRank',
                                          'Mas_BestRank': 'LTeam_Mas_BestRank',
                                          'Mas_WorstRank': 'LTeam_Mas_WorstRank',
                                          
                                          'Rpi_AvgRank': 'LTeam_Rpi_AvgRank',
                                          'Rpi_BestRank': 'LTeam_Rpi_BestRank',
                                          'Rpi_WorstRank': 'LTeam_Rpi_WorstRank',
                                          
                                          'exp_coach': 'LTeam_Exp_Coach',
                                          
                                          'rolling_win_percentage': 'LTeam_rolling_win_percentage',
                                          'season_last10_wins_percentage': 'LTeam_season_last10_wins_percentage',
                                          'last10_consecutive_wins': 'LTeam_last10_consecutive_wins',
                                          
                                          'WinPct': 'LTeam_WinPct',
                                          'FG_ratio' : 'LTeam_FG_ratio',
                                          '3PT_ratio': 'LTeam_3PT_ratio',
                                          'ATO_ratio': 'LTeam_ATO_ratio'}, inplace = True)


# ------------
# Recode the Winner's Location Variable (WLoc)
# ------------
games_with_season_stats['WLoc'] = games_with_season_stats['WLoc'].replace(['N', 'H', 'A'], [0, 1, -1])


# ------------
# Final All-Game File
# ------------
games_with_season_stats.to_csv('/Users/mfeene/Desktop/marchmadness_2024/games_with_season_stats_addl.csv')