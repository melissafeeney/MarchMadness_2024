#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:14:08 2024

@author: mfeene
"""

# -------------------------
# IV. GENERATE MATCHUP FILE
# -------------------------

import pandas as pd
import itertools

# Generate file with all possible matchups
path = '/Users/mfeene/Desktop/marchmadness_2024/kaggle_data_31824/'
regseason = pd.read_csv(path + 'MRegularSeasonDetailedResults.csv')
teams = pd.read_csv(path + 'MTeams.csv')

# All the teams who are D1 in 2024 season
teams_2024 = teams[teams['LastD1Season'] == 2024]['TeamID'].values

matchups = itertools.combinations(teams_2024, 2)

matchups_list = []
for matchup in matchups:
    #print(matchup)
    matchups_list.append(matchup)
    
sample_submission = pd.DataFrame(matchups_list)
sample_submission.columns = ['team1', 'team2']
sample_submission['season'] = 2024 

sample_submission['ID'] = sample_submission['season'].astype(str) + '_' + sample_submission['team1'].astype(str) + '_' + sample_submission['team2'].astype(str)
sample_submission['Pred'] = 0.5

del sample_submission['season']
del sample_submission['team1']
del sample_submission['team2']

sample_submission.to_csv('SampleSubmission2024.csv')