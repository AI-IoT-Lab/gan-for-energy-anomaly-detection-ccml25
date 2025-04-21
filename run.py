import pandas as pd
import subprocess
import json
import time
import sys

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="run.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)



# To perform all the operations (-one model per building)
with open('config.json', 'r') as file:
    config = json.load(file)

# open and get all the unique buildings:
df = pd.read_csv(config["data"]["dataset_path"])
uni_b = df["building_id"].unique()
logger.info(f"unique builds : {uni_b}")
l = len(uni_b)
i = 1  # counter for builds

for b_id in df["building_id"].unique():
    logger.info(b_id)
    # Change configs
    config["data"]["only_building"] = int(b_id)

    with open('config.json', 'w') as file:
        json.dump(config, file)

    start_time = time.time()
    subprocess.run([sys.executable, "preprocessing.py"])
    end_time = time.time()
    pre_time = end_time - start_time
    logger.info(f"Building {b_id} :: Time taken for preprocessing : {pre_time}")

    starting_time = time.time()
    subprocess.run([sys.executable, "training.py"])
    end_time = time.time()
    train_time = end_time - starting_time
    logger.info(f"Building {b_id} :: Time taken for training : {train_time}")

    start_time = time.time()
    subprocess.run([sys.executable, "test.py"])
    end_time = time.time()
    test_time = end_time - start_time
    logger.info(f"Building {b_id} :: Time taken for test : {test_time}")
    logger.info(f"Processed {i}/{l} buildings ...... Total time taken: {end_time - starting_time} secs")
    i = i + 1
