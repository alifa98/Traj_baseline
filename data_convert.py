# this script converts higher order hexagon data to the format of the Libcity model
# Output is a dyna file with the following format:
# dyna_id,type,time,entity_id,location

import os
import pandas as pd
import tqdm

input_file = input_file = '/local/data1/shared_data/higher_order_trajectory/geolife/ho_geolife_res7.csv'
dataset_name = 'geolife_res7'
user_id_column = 'user_id'


output_folder = 'raw_data/{}/'.format(dataset_name)
output_file = output_folder + '{}.dyna'.format(dataset_name)

os.makedirs(output_folder, exist_ok=True)

# read input file
input_file = pd.read_csv(input_file)

output_columns = ['dyna_id', 'type', 'time', 'entity_id', 'location']
record_id = 0
max_time = 0
with open(output_file, 'w') as file:
    file.write(','.join(output_columns) + '\n')
    for index, row in tqdm.tqdm(input_file.iterrows(), desc="Converting data"):
        for time_step, loc_hex in enumerate(row['higher_order_trajectory'].split(' ')):
            file.write(
                f'{record_id},trajectory,{time_step},{row[user_id_column]},{loc_hex}\n')
            record_id += 1
            max_time = max(max_time, time_step)
        file.flush()

with open(f"{output_folder}{dataset_name}-info.txt", "w") as f:
    f.write(f"max_time:{max_time}\n")
