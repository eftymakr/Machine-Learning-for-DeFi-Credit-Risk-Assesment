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


2. **Machine Learning:**
- Apply train/test split, oversampling, statistical learning and deep learning techniques by using the file `train_ensemble_models.py` or `train_CNN_model.py` and `Stat_learning_set_v0x.csv` as an input file. 


