import json
import pandas as pd
import numpy as np
from preprocess import split_sequence
import  pickle
prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="test.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# configs
nz = config['training']['latent_dim']
window_size = config['preprocessing']['window_size']
iters = config["recon"]["iters"]
b_id = "all"
if config['data']["only_building"] is not None:
    b_id = config['data']["only_building"]

# model/data import

with open(f"IForest_{b_id}","rb") as f:
    clf = pickle.load(f)

test_df = pd.read_csv(f"test_df_{b_id}.csv")
train_df = pd.read_csv(f"train_df_{b_id}.csv")

# testing
temp = test_df.groupby("s_no")  # group by segments
n_segs = temp.ngroups  # total number of segments
# get the segment no.s present in file
test_seg_ids = test_df["s_no"].unique()
train_seg_ids = train_df["s_no"].unique()
logger.info(f"Total segments : {n_segs} ")

# storing reconstruction details ...
test_out = {}
for id, id_df in temp:
    id_out = {"X": None, "Z": None, "X_": None, "labels": None, "window_b_included": False,
              "window_a_included": False}
    id_df.reset_index(drop=True, inplace=True)
    segment = np.array(id_df["meter_reading"])
    # add window length from segment before in front (if available)
    before_id = id - 1
    if before_id in test_seg_ids:
        b = test_df[test_df["s_no"] == before_id]["meter_reading"][-window_size//2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    if before_id in train_seg_ids:
        b = train_df[train_df["s_no"] == before_id]["meter_reading"][-window_size//2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    # add window length from segment after at back (if available)
    after_id = id + 1
    if after_id in test_seg_ids:
        a = test_df[test_df["s_no"] == after_id]["meter_reading"][:window_size//2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    if after_id in train_seg_ids:
        a = train_df[train_df["s_no"] == after_id]["meter_reading"][:window_size//2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    logger.info(f'"diff in length :", {len(id_df) - len(segment)}, {( id_out["window_b_included"],id_out["window_a_included"])}')
    # each segment will have subsequences of overlapping windows:
    X = split_sequence(segment, window_size)
    id_out["X"] = X
    # reconstruct & errors
    loss_list = clf.predict(X)
    id_out["loss"] = loss_list
    test_out[id] = id_out

# Store the dict as pickle
with open(f'loss_{b_id}.pkl', 'wb') as file:
    pickle.dump(test_out, file)
