Dynamic Outcomes Based Clustering
===============================

This repository contains the code used for **Dynamic Outcomes-Based Clustering of Disease Trajectory in Mechanically Ventilated Patients** and implementation instructions. You can watch a brief project talk here:

[![Watch the video](https://img.youtube.com/vi/06EQ5Xzal-0/maxresdefault.jpg)](https://www.youtube.com/watch?v=06EQ5Xzal-0)
 
## Citation
If you use this code or the models in your research, please cite the following:

```
@inproceedings{
    rocheteau2022dynamic,
    title={Dynamic Outcomes-Based Clustering of Disease Trajectory in Mechanically Ventilated Patients},
    author={Emma Charlotte Rocheteau and Ioana Bica and Pietro Lio and Ari Ercole},
    booktitle={NeurIPS 2022 Workshop on Learning from Time Series for Health},
    year={2022},
    url={https://openreview.net/forum?id=S7FEB6rwc5R}
}
```

## Motivation
The advancement of Electronic Health Records (EHRs) and machine learning have enabled a data-driven and personalised approach to healthcare. One step in this direction is to uncover patient sub-types with similar disease trajectories in a heterogeneous population. This is especially important in the context of mechanical ventilation in intensive care, where mortality is high and there is no consensus on treatment. In this work, we present an approach to clustering mechanical ventilation episodes, using a multi-task combination of supervised, self-supervised and unsupervised learning techniques. Our dynamic clustering assignment is guided to reflect the phenotype, trajectory and outcomes of the patient. Experimentation on a real-world dataset is encouraging, and we hope that this could translate into actionable insights in guiding future clinical research.

## Headline Results

TBD

## Pre-processing Instructions

1) To run the sql files you must have the AmsterdamUMCdb database set up: https://amsterdammedicaldatascience.nl/amsterdamumcdb.

2) Follow the instructions: https://github.com/AmsterdamUMC/AmsterdamUMCdb to ensure the correct connection configuration. 

3) Replace the amsterdam_path in `paths.json` to a convenient location in your computer, and do the same for `amsterdamUMCdb_preprocessing/create_all_tables.sql` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/'`. Leave the extra '/' at the end.

4) In your terminal, navigate to the project directory, then type the following commands:

    ```
    psql 'dbname=amsterdam user=amsterdam options=--search_path=amsterdam'
    ```
    
    Inside the psql console:
    
    ```
    \i amsterdamUMCdb_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m amsterdamUMCdb_preprocessing.run_all_preprocessing
    ```
    
It will create the following directory structure:
   
```bash
MIMIC_data
├── test
│   ├── data.npz
│   └── vents.txt
├── train
│   ├── data.npz
│   └── vents.txt
├── val
│   ├── data.npz
│   └── vents.txt
├── flat_features.csv
├── labels.csv
├── standardisation_limits.csv
├── timeseries.csv
├── ventilator_settings.csv
└── vents.txt

```
    
   
## Running the models
Once you have run the pre-processing steps you can run all the models in your terminal. Set the working directory to Mechanical-Ventilation-Clustering, and run the following:

```
python3 -m train_prediction_only
```

```
python3 -m train_prediction_reconstruction
```
    
Note that your experiment can be customised by using command line arguments (these can be explored in args.py).
    
Each experiment you run will create a directory within logs. The naming of the directory is based on 
the date and time that you ran the experiment (to ensure that there are no name clashes). 

