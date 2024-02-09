# Machine-Learning-for-DeFi-Credit-Risk-Assesment
Utilizing a diverse dataset from blockchains like Ethereum, with data preprocessing, we extract DeFi-specific features for transaction statistics. We apply machine learning models (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, CNN) to predict wallet liquidations, using accuracy and AUC for evaluation.

### `Output_files` Directory
- This directory holds output files generated during the processing.

## Preprocessing and Learning Process

1. **Data Preprocessing:**
   - Use `preprocessing_filtering_v01.py` to preprocess data and apply cuts (at least 1 borrow event, and at least 10 transactions). The arguments are "input file", stop_chunks (when to stop, 5400 total, so input 10,000 to get all), and skip_chunks (skip chunks in the beginning, set to 0 not to skip anything)
       - Input file: `Lending_export.csv`, to be provided via argument.
       - Output file 1: `stats_lending_dataset_v0x.csv`, which contains aggregated statistics on the lending dataset.
       - Output file 2: `borrowed_10txns_good_list_v0x.csv`, which contains all addresses that fulfil the fitler criteria.
       - Output file 3: `full_training_set_borrowed_10txns_part{i}_v0x.csv`, which contains the full feature space for all addresses that fit the cuts, split up in chunks of 10,000.

   - Further process data with `preprocessing_time_aggregation_v0x.py` to aggregate information based on time. The arguments are "split_counter"; if all data was processed then this is 9.
       - Input files: `full_training_set_borrowed_10txns_part{i}_v0x.csv`, which contains the full feature space for all addresses that fit the cuts, split up in chunks of 10,000.
       - Output file: `ML_input_v0x.csv`, which contains the training data aggregated my month, including the flag for whether the address was liquidated.
         
   - The file `preprocessing_pivot_v01.py` pivots the temporal component of the dataset.
       - Input file: `ML_input_v0x.csv`, which contains the training data aggregated my month, including the flag for whether the address was liquidated.
       - Output file 1: `Stat_learning_set_v0x.csv`, which contains the same data as `ML_input_v0x.csv`, but the time steps are pivoted out, hence it only contains one record per wallet address.

   - The file `preprocessing_oversampling_v03.py` oversamples the data and stores individual files that are to be used for the training.
       - Input file: `Stat_learning_set_v0x.csv`, which contains the same data as `ML_input_v0x.csv`, but the time steps are pivoted out, hence it only contains one record per wallet address.
       - Output file 2: `X_train_chunk_{i}.npy` and `y_train_chunk_{i}.npy`, these contain oversampled training data of the `Stat_learning_set_v0x.csv` file
       - Output file 2: `data_test.npy` and `full_data.npy`, these contains the original data from `Stat_learning_set_v0x.csv` in a file format for training.


3. **Machine Learning:**
- Apply statistical learning and deep learning techniques using `statistical_learning_v05.py`, passing the model; options are "XGB", "CatBoost", "LightGBM", "RF", "LR", and "CNN".



## Usage

1. Run the data preprocessing scripts (`preprocessing_filtering_v01.py` and `preprocessing_time_aggregation_v01.py`) to prepare the data.

2. For statistical learning approach:
   - Execute `statistical_learning_v05.py` for applying statistical learning models.
