# ML_project

## Abstract

Our project will identify umpire bias in baseball by analyzing pitch 
location and outcomes by pitch. We want to understand what makes an umpire 
more accurate at calling strikes through situational outcomes and player
demographics, including team, race and home country. This is a binary 
classification problem and we will be looking at data with a lot of features
and trying to figure out which of these attributes effect bias the most. 
We propose to evaluate our success by subsetting our data and testing it 
to see the true accuracy of our model on actual game data.

## Motivation

Robot umpires are in the hole. If actual umpires strike out, it will be their
time to bat. If we have a successful model, we can understand what attributes
cause umpires to be more biased. The more successful the model is, the more umpire 
bias will be evident.

## Planned Deliverables

Using data that includes information about the physics of the pitch such as launch speed, direction, and angle as well as information about the current state of the game we will develop a model that predicts whether or not the umpire will make the correct call; we have the information of the actual location of the ball as it crosses home plate, so we know what the call should of been and we also have the call the umpire makes. Thus, we will develop a model that predicts whether or not the umpire will make the correct call. A majority of the work for our project will be in analyzing partitions of the data to explore factors which may affect the umpire's ability to make the correct call, such as pressure in late game scenarios or in demographic bias towards the batter or the pitcher. Of course, pitches on the outside of the box will be less likely to be called correctly but there may be other nuances in the model that could be interesting to explore. This could also give insight into times when it might be useful to employ an automatic pitch calling system in situations when umpires are less likely to make the correct call.  

## Resources Required

We will require a few different datasets for this project. The first one is [here][https://www.kaggle.com/datasets/amandaaapoor/2021regularseasonmlbpitches]. This dataset includes all of the 700,000 pitches thrown in the MLB regular season in 2021. It includes has about the pitch (speed, location over the plate, what zone it was in over the plate, whether it was called a strike or a ball, etc). It also has information about the pitcher and the batter for each pitch, identified by their ID #. We can use the [MLB API] [https://appac.github.io/mlb-data-api-docs/#header-getting-started] to pull a dataset of player info, which includes name variants, education information, country of origin and attributes like height, weight and age. We can merge this dataset with the pitch dataset by player ID to include information about the demographics of the pitchers and batters in the pitch dataset. 

## What We will Learn

We aim to gain insights into the factors that influence umpire decision-making and the presence of potential bias in strike zone calls. By analyzing pitch characteristics, game situations, and player demographics, we hope to uncover patterns that reveal when and why umpires are more likely to make incorrect calls. Ideally, we could determine the ways in which cause umpire's unfairly call pitches or potentially how miscalled pitches can alter the outcome of games. 

## Risk Statement

There are a few risks that might be associated with this project. First off, we do have a large dataset and might require significant computational power or being clever about how we subset the data so that we can use a smaller portion of it. We also might not find any patterns ‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍‍in the data or be able to predict whether a pitch is called correctly.  ‍‍‍‍‍‍‍‍‍‍‍‍‍‍We may find that umpires call a pitch correctly at the same rate regardless of game situation, player demographics, or point in the game. Even if we aren't able to predict accurately whether a pitch will be called correctly, this situation could still give insight into whether there are any patterns in when umpires call a pitch correctly. 

## Ethics Statement

Umpires, players and baseball fans and executives have the potential to benefit.
At the same time, if umpires are biased, they would be harmed because of this analysis
because they could be at risk of replacement. The world will be an overall better place
because we will be able to understand and limit bias in the world at baseball. 
- Missed calls by umpires are predictable by features including pitch location, race, team and game
situation.
-The world is a better place when baseball games are called more accuratley and 
umpires miss fewer calls. 

## Tentative Timeline
**End of three-week check in:**
- Have all data downloaded and cleaned/preprocessed
- Have some exploratory analysis done that includes some plots which will help us figure out which features we want to focus on
- Have an idea of what type of model we want to use for our classification 
**Last couple of weeks**
- Implement the model we decided on for classification 
- Audit the model for bias using different features in the dataset (point in game, player demographics, game situation, etc.)
- Create presentation