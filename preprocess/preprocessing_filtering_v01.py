import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
#import tqdm.notebook as tq

def process_lending_data(input_file, stop_iteration, skip_chunks):
    # Read the lending dataset in chunks. This contains all relevant protocols except Balancer. 
    # There is another dataset in the same format that only contains Balancer data, however, this has not been used so far.
    data_chunks = pd.read_csv(input_file, chunksize=10000, lineterminator='\n')

    # Initialize variables for chunk iteration and aggregate statistics.
    chunk_iterator = 0
    aggregate_stats = pd.DataFrame()

    # Iterate through data chunks and aggregate statistics
    for chunk in data_chunks:
        print('processing chunk', chunk_iterator)
        chunk_iterator += 1

        # Group data around address, protocol, chain, and event name
        if chunk_iterator == 1:
            chunk_summary = chunk.groupby(['protocol_name', 'event', 'chain_name', 'user']).count()
            aggregate_stats = chunk_summary[['protocol_address']]
        # Skip n chunks if specified
        elif chunk_iterator < skip_chunks:
            continue
        else:
            chunk_summary = chunk.groupby(['protocol_name', 'event', 'chain_name', 'user']).count()
            aggregate_stats = aggregate_stats.add(chunk_summary[['protocol_address']], fill_value=0)


        # Break early for testing or if running on a low resource machine
        if chunk_iterator == stop_iteration:
            break

    # Save aggregated statistics to a CSV file
    aggregate_stats.to_csv('Output_files/stats_lending_dataset_v03.csv')

    # Read the aggregated statistics from the CSV file
    stats_df = pd.read_csv('Output_files/stats_lending_dataset_v03.csv')

    # Rename columns for clarity
    stats_df = stats_df.rename(columns={'user': 'address', 'protocol_address': 'num_transactions'})

    # Reshape the data for analysis
    grouped_df = stats_df.groupby(['address', 'protocol_name'], as_index=False).sum(numeric_only=False)
    pivoted_df = grouped_df.pivot(index='address', columns='protocol_name', values='num_transactions')
    pivoted_df.reset_index(inplace=True)
    pivoted_df = pivoted_df.fillna(0)

    # Group and count interactions by protocol name
    protocol_interaction_count = stats_df[['address', 'protocol_name']].groupby(['address', 'protocol_name'], as_index=False).sum().groupby(['address'], as_index=False).count()
    protocol_interaction_count.sort_values('protocol_name')
    protocol_plot_data = protocol_interaction_count.groupby('protocol_name', as_index=False).count()


    # Create a bar plot for protocol interactions
    plt.bar(protocol_plot_data.protocol_name, protocol_plot_data.address, log=True)
    plt.xticks(range(1, len(protocol_plot_data) + 2, 1))
    plt.title('Number of wallets by multiprotocol interaction')
    plt.xlim(1, len(protocol_plot_data) + 1)
    plt.show()

    # Group and count interactions by chain name
    
    chain_interaction_count = stats_df[['address', 'chain_name']].groupby(['address', 'chain_name'], as_index=False).sum().groupby(['address'], as_index=False).count()
    chain_interaction_count.sort_values('chain_name')
    chain_plot_data = chain_interaction_count.groupby('chain_name', as_index=False).count()

    # Create a bar plot for chain interactions
    
    plt.bar(chain_plot_data.chain_name, chain_plot_data.address, log=True)
    plt.xticks(range(1, len(chain_plot_data) + 2, 1))
    plt.title('Number of wallets by multichain interaction')
    plt.xlim(1, len(chain_plot_data) + 1)
    plt.show()

    # Identify number of borrowed transactions and create a list of 'good' borrowers, i.e., that have at least one borrow event
    stats_df['borrowed'] = stats_df.apply(lambda x: 1 if (x['event'] == 'borrow' and x['num_transactions'] >= 0) else 0, axis=1)
    good_borrowers = stats_df[stats_df['borrowed'] == 1].groupby('address', as_index=False).count()

    # Calculate statistics
    total_wallets = len(stats_df.groupby('address').count())
    num_good_borrowers = len(good_borrowers)
    percentage_good_borrowers = num_good_borrowers / total_wallets
    print('Percent of wallets that have at least 10 transactions and 1 borrow event:', percentage_good_borrowers)

    # Save the list of good borrowers to a CSV file
    good_borrower_list = good_borrowers.address.to_list()
    address_stats = stats_df.groupby('address', as_index=False).sum(numeric_only=False)
    address_stats_with_min_transactions = address_stats[address_stats.num_transactions >= 10]
    address_stats_with_min_transactions_filtered = address_stats_with_min_transactions[address_stats_with_min_transactions.num_transactions > 10]
    address_stats_with_min_transactions_filtered['good'] = address_stats_with_min_transactions_filtered.apply(lambda x: 1 if (x['address'] in good_borrower_list) else 0, axis=1)
    address_stats_with_min_transactions_filtered.to_csv('Output_files/borrowed_10txns_good_list_v02.csv')

    # Load address data and create a DataFrame from the lending export CSV
    date_to_number = {}
    first_date = datetime(2019, 5, 1)
    end_date = datetime(2023, 6, 1)
    delta = timedelta(days=30)
    iterator = 0

    # Create a mapping of dates to numbers for time-based analysis
    for year in range(2019, 2024):
        for month in range(1, 13):
            date_str = f'{year}-{str(month).zfill(2)}'
            current_date = datetime(year, month, 1)
            if current_date < first_date:
                continue
            date_to_number[date_str] = iterator
            iterator += 1

    # Load the list of good addresses and lending data chunks
    good_address_list = pd.read_csv('Output_files/borrowed_10txns_good_list_v02.csv').address.to_list()
    data_chunks = pd.read_csv(input_file, chunksize=10000, lineterminator='\n')

    # Start processing and filtering data
    start_time = time.time()
    good_address_set = set(good_address_list)
    df_train_list = []
    iterator = 0
    split_counter = 0

    # Iterate through data chunks and filter for good addresses
    for chunk in data_chunks:
        #print('processing chunk', chunk_iterator)
        if chunk_iterator < skip_chunks:
            continue

        data_filtered = chunk[chunk['user'].isin(good_address_set)].copy()
        data_filtered['YearMonth'] = -999
        data_filtered['YearMonth'] = data_filtered.apply(lambda x: x['signed_at'][:7], axis=1)
        data_filtered = data_filtered.replace({"YearMonth": date_to_number})
        data_filtered['address'] = data_filtered.user

        #df_train_list.append(data_filtered[['address', 'YearMonth', 'event', 'amount_in_usd', 'amount_out_usd']])
        df_train_list.append(data_filtered[['address', 'protocol_address','YearMonth', 'event', 'amount_in_usd', 'amount_out_usd']])

        iterator += 1
    #     print('Processed', iterator, 'chunks,', sum(len(df) for df in df_train_list), 'txns found,', int((time.time() - start_time)), 'seconds')
        if iterator % 500 == 0:
            print('Processed', iterator, 'chunks,', sum(len(df) for df in df_train_list), 'txns found,', int((time.time() - start_time)), 'seconds')
            df_train = pd.concat(df_train_list)
            df_train.reset_index(drop=True, inplace=True)
            df_train.to_csv('full_training_set_borrowed_10txns_part'+str(split_counter)+'_v02.csv', index=False)
            split_counter += 1
            df_train_list = []

        if iterator == stop_iteration:
            print('Processed', iterator, 'chunks,', sum(len(df) for df in df_train_list), 'txns found,', int((time.time() - start_time)), 'seconds')
            df_train = pd.concat(df_train_list)
            df_train.reset_index(drop=True, inplace=True)
            df_train.to_csv('full_training_set_borrowed_10txns_part'+str(split_counter)+'_v02.csv', index=False)
            break  

    print('Done!')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file_path stop_iteration skip_chunks")
        sys.exit(1)
    
    input_file = sys.argv[1]
    stop_iteration = int(sys.argv[2])
    skip_chunks = int(sys.argv[3])
    process_lending_data(input_file, stop_iteration, skip_chunks)
    print('Done!')
