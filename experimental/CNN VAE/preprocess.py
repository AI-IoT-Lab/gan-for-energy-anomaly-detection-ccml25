import pandas as pd
import numpy as np
import json
import torch
import os
import sys

from preprocessing import preprocess_data,segment_data,split_sequence

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="preprocessing.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    prefix = "../../"
    # Load config
    with open(prefix + 'config.json', 'r') as file:
        config = json.load(file)

    window_size = config['preprocessing']['window_size']
    b_id = "all"

    # train-test segments
    data = pd.read_csv(prefix+f"{config['data']['dataset_path']}")
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]   # one particular building at a time (recommended)
        data = data[data["building_id"] == b_id]

    data = preprocess_data(data)
    train_df, test_df, s_no, min_len = segment_data(data,normalize=True)
    logger.info(f"total number of segments : {s_no}")
    logger.info(f" min len of segment: {min_len}")

    # storing segments
    train_df.to_csv(f"train_df_{b_id}.csv", index=False)
    test_df.to_csv(f"test_df_{b_id}.csv", index=False)  # will be used for testing later

    # Convert training data into model input:
    X_train = []
    seg_count = 0
    temp = train_df.groupby("s_no")
    for id, id_df in temp:
        X_w = split_sequence(id_df["meter_reading"], window_size)
        X_train.extend(X_w)
        seg_count += 1
    X_train = np.array(X_train)
    X_train = X_train.reshape(len(X_train), 1, -1)

    logger.info(f"training tensor shape: {X_train.shape}")
    torch.save(X_train, f"X_train_{b_id}.pt" )
    logger.info(f'The model training input is stored at : {os.getcwd()}')
