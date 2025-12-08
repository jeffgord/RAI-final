This repository contains code for training and evaluating music recommendation models, focusing on fairness analysis across artist genders. The implementation is based on the [1st place solution](https://github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution) from the WSDM CUP 2018 Music Recommendation Challenge.

Description of the various files:

- `analysis-final.ipynb` - final profiling and analysis


- `analysis-scratch.ipynb` - temporary notebook used for initial analysis of results


- `artist-scraper.ipynb` - used to grab artist demographic data from MusicBrainz


- `prediction-merge.ipynb` - merges predictions from the individual models to create a set of final predictions


- `profiling.ipynb` - temporary notebook used for initial profiling analyses
  

## Data Requirements

To run various notebooks, you will need to add the following data files (from [Kaggle Competition](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)) manually to the `data/` folder:

- `members.csv`
- `song_extra_info.csv`
- `songs.csv`
- `train.csv`

These data files are not stored in git because of their size.

Additionally, the `artist-scraper.ipynb` notebook will create:
- `artists.csv` (contains artist gender and country information)

## Project Structure

```
RAI-final/
├── data/
│   ├── predictions/
│   │   ├── lgb/          
│   │   ├── nn/           
│   │   └── final/        
│   ├── artists.csv
│   ├── songs.csv
│   ├── train.csv
│   └── ...
├── scripts/
│   ├── lgb_training_multi_seed.py    # LightGBM training script
│   ├── nn_training_multi_seed.py     # Neural Network training script
│   ├── nn_generator.py               # Data generator for NN training
│   ├── lgb_run.sbatch                # batch script for LGB
│   ├── nn_run.sbatch                 # batch script for NN
│   └── old_training_scripts/         
├── analysis-scratch.ipynb            
├── analysis-final.ipynb              # Final analysis notebook
├── prediction-merge.ipynb            # Merges LGB and NN predictions
└── README.md
```

## Running Predictions

The prediction pipeline consists of three main steps:

### Step 1: Train LightGBM Models

**Using a batch script:**

```bash
cd scripts
sbatch lgb_run.sbatch
```

**Or run directly:**

```bash
cd scripts
python lgb_training_multi_seed.py
```

This script will:
- Train 10 LightGBM models with random seeds
- Save predictions to `data/predictions/lgb/lgb_<auc>_seed<seed>.csv`
- Generate a summary file at `temp_lgb/multi_seed_summary.csv`

**Expected Output:**
- 10 prediction files in `data/predictions/lgb/` (one per seed)

### Step 2: Train Neural Network Models

Train neural network models with the same random seeds for ensemble consistency.

**Using a batch script:**

```bash
cd scripts
sbatch nn_run.sbatch
```

**Or run directly:**

```bash
cd scripts
python nn_training_multi_seed.py
```

This script will:
- Train 10 neural network models with random seeds
- Save predictions to `data/predictions/nn/nn_<auc>_<loss>_seed<seed>.csv`
- Generate a summary file at `temp_nn/multi_seed_summary.csv`

**Expected Output:**
- 10 prediction files in `data/predictions/nn/` (one per seed)

### Step 3: Merge Predictions

Combine LightGBM and Neural Network predictions using ensemble weighting.

Use `prediction-merge.ipynb` to:

- Load corresponding LGB and NN predictions
- Merge them using weighted ensemble: `final_pred = 0.6 * lgb_pred + 0.4 * nn_pred`
- Save final predictions to `data/predictions/final/final_seed<seed>.csv`

**Ensemble Weighting:**
The 0.6/0.4 weighting follows the approach from the [1st place solution](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/discussion/45942).

**Expected Output:**
- 10 final prediction files in `data/predictions/final/` (one per seed)

## Running Analysis

After generating predictions, run the analysis notebooks:

**`analysis-final.ipynb`**

## System Requirements

### Dependencies:
- pandas >= 2.0.3
- numpy >= 1.24.4
- scikit-learn >= 1.2.1
- lightgbm
- tensorflow (with GPU support for NN training)
- keras
- fairlearn (for fairness metrics)
- matplotlib (for visualizations)

### Batch Configuration:
The batch scripts are configured for HPC environments with:
- Singularity containers
- GPU access (for NN training)
- Sufficient memory and CPU allocation
