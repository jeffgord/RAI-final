These notebooks support our analysis conducted on the winning solution to the KKBox Kaggle music recommendation comeptition:

https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data

Here is a discription of the various files:
- analysis-final.ipynb - final profiling and analysis
- analysis-scratch.ipynb - temporary notebook used for initial analysis of results
- artist-scraper.ipynb - used to grab artist demographic data from MusicBrainz
- prediction-merge.ipynb - merges predictions from the individual models to create a set of final predictions
- profiling.ipynb - temporary notebook used for initial profiling analyses

Note that to run various notebooks, you will need to add the following data files from the competition to the data/ folder:
- members.csv
- song_extra_info.csv
- songs.csv
- train.csv

These data files are not stored in git because of their large size.

The artist-scraper.ipynb notebook will create a new data file with artist demographic information:
- artists.csv

Scripts include our modifications to the original model training and prediction code that were necessary for the purposes of our analysis.
