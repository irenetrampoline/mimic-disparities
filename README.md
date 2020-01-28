# Disparities in MIMIC-III

Here we recreate plots from ["Can AI Hep Reduce Disparities in General Medical and Mental Health Care?"](https://journalofethics.ama-assn.org/article/can-ai-help-reduce-disparities-general-medical-and-mental-health-care/2019-02) by Chen, Szolovits, and Ghassemi 2019, (AMA Journal of Ethics)

Because of data proprietary reasons, we cannot share results from the psychiatric dataset. 

We demonstrate:
 1. Data hetereogeneity in the MIMIC clinical notes through LDA topic modeling and disparities in topics by race, gender, and insurance type
 2. Disparities in predictive accuracy by race, gender, and insurance type

## Recreating results

1) Get MIMIC notes from `make_mimic_notes.py`. You will need to adjust the username and location of MIMIC data. 

2) Get Mallet topics from the notes. We convert the notes into separate text files in `make_mallet_data.py`. We then run Mallet in `run_mallet_topics.sh`.

3) Create plots in `Recreate_Plots.ipynb`

## Requirements

1) [MIMIC data access](https://mimic.physionet.org/gettingstarted/access/)
2) Download [Mallet](http://mallet.cs.umass.edu/download.php)
3) Python packages listed in `requirements.txt`
