-- creates all the tables and produces csv files
-- takes a little while to run (approx 30 mins to an hour)

-- change the paths to those in your local computer using find and replace for '/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/'.
-- keep the file names the same

\i amsterdamUMCdb_preprocessing/labels.sql
\i amsterdamUMCdb_preprocessing/flat_features.sql
\i amsterdamUMCdb_preprocessing/timeseries.sql

\copy (select * from labels) to '/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/labels.csv' with csv header
\copy (select * from ventilationsettings) to '/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/ventilator_settings.csv' with csv header
\copy (select * from flat) to '/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/flat_features.csv' with csv header
\copy (select * from timeseries) to '/Users/emmarocheteau/PycharmProjects/Mechanical-Ventilation-Clustering/amsterdam_data/timeseries.csv' with csv header