AmsterdamUMCdb pre-processing
==================================

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