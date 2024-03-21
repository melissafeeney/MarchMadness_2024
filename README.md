## March Madness 2024

Happy March! It's that time of year again that raises the question- can a machine learning algorithm design a successful 2024 March Madness tournament bracket?

This year, I decided to combine two powerful advancements in machine learning into my algorithm design process- attention mechanisms and model blending. I trained my models on every game from 2003 through 2024 (pre-tournament), amounting to over 130K games. 

The attention mechanism is a pivotal advancement in deep learning introduced in 2015, which encourages focus on the most important elements of an input sequence to a task at hand. It mimics how humans focus- we concentrate on important details and filter out distractions. For example, think of a time when you were in a crowded and noisy room, but concentrated on a single conversation you were having with someone. Across 130K games, there is likely a lot of noise to filter out!

Model blending is a technique that takes the outputs of multiple independently trained submodels and feeds them into a final model, based on the belief that such a cascading structure will result in better performance.

As you can see in my brackets, my algorithms assign probabilities to the winner of each game. For example, it is not surprising that in games featuring wider skill gaps between teams, such as the first-round game of Connecticut (1 seed) vs. Stetson (16 seed), Connecticut is given over a 96% chance of winning. While games featuring closer-seeded teams such as those between 8 and 9 seeds have less decisive win probabilities- often closer to 55-65%. As the tournament progresses, and the seed differences between teams become smaller, the games presumably become more competitive and the probabilities are even closer to 50%, indicating that those games are likely more of a toss-up- and more exciting!

### My process looked like this:
1. Blend 5 independently trained neural networks that do not use attention mechanisms into one final neural network
2. Blend 5 independently trained neural networks that use attention mechanisms into one final neural network with an attention mechanism

### Feature engineering 
A critical part of this exercise, I decided upon 20 independent variables for each team:
- strength of schedule proxies determined by KenPom, Massey, and RPI rankings, 
- season-level stats including coach seniority, team wins and losses, and ratios representing % of field goals made, % of 3-pointers made, and assists to turnovers
- rolling 10-game win percentage for each team through each season
- considering the last 10 games of the regular season played by each team to identify "hot streaks"; their win percentage and consecutive wins in those last 10 games

### Model Architecture

Model without the attention mechanism:

![model_graph](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/c92660eb-7077-4b06-adac-652af6dc7bec)

Model with the attention mechanism:

![model_graph](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/1583d088-784b-4f48-9c2b-882049959992)

### The Brackets!

Model without the attention mechanism:

![non_attention_MNCAA2024](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/94f6cc31-91b5-4d80-84ef-cba5ab123ae8)


Model with the attention mechanism:

![attention_MNCAA2024](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/22c0fac7-9aea-4fc1-9093-c66bfbb09eae)


### Model Results on Test Data
Model without the attention mechanism:

![results](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/cb2ab046-99a2-4649-a5f1-905bfa0e1ca0)

Model with the attention mechanism:

![results](https://github.com/melissafeeney/MarchMadness_2024/assets/31778500/7ca3dda2-32a8-4de2-885e-5c91e949a445)

### Results Discussion
The test set results of both the architecture without the attention mechanism and that with the attention mechanism are very similar- all exceeding 75% in accuracy, F1 score, ROC AUC, precision, and recall. The classification reports indicate the strong abilities of both architectures to identify wins (1.0) and losses (0.0) of teams- there is not a bias toward either way. 

### Other Thoughts
The model without the attention mechanism picked Connecticut to win it all, with the championship game featuring Connecticut and Houston. The model with the attention mechanism picked Purdue to win it all, with the championship game featuring Purdue and North Carolina. With Connecticut being last year's champion and a common favorite this year, it does not surprise me that the model without the attention mechanism and the model with the attention mechanism picked different champions. With the entire premise of the attention mechanism being to sift through the noise to identify signals, it does not surprise me that that model did not select the common favorite- instead choosing to follow more nuanced patterns in the data and selecting Purdue instead. 
