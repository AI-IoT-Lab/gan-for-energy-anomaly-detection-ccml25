import pandas as pd
import torch
import numpy as np
from preprocess import split_sequence
import pickle
import json
import torch.nn as nn
from reconstruction import reconstruct

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="test.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)



prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)



# configs
nz = config['training']['latent_dim']
window_size = config['preprocessing']['window_size']
b_id = "all"
if config['data']["only_building"] is not None:
    b_id = config['data']["only_building"]


# model/data import
vae= torch.load(f'vae_{b_id}.pth')
logger.info(f"vae imported : vae_{b_id}.pth")
vae.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(device)
test_df = pd.read_csv(f"test_df_{b_id}.csv")
train_df = pd.read_csv(f"train_df_{b_id}.csv")


# testing
temp = test_df.groupby("s_no")  # group by segments
n_segs = temp.ngroups  # total number of segments
# get the segment no.s present in file
test_seg_ids = test_df["s_no"].unique()
train_seg_ids = train_df["s_no"].unique()
logger.info(f"Total segments : {n_segs} ")
criterion = nn.MSELoss(reduction='none').to(device)
# storing reconstruction details ...
test_out = {}
for id, id_df in temp:
    id_out = {"X": None, "Z": None, "X_": None, "recon_loss": None, "labels": None, "window_b_included": False,
              "window_a_included": False}
    id_df.reset_index(drop=True, inplace=True)
    segment = np.array(id_df["meter_reading"])
    # add window length from segment before in front (if available)
    before_id = id - 1
    if before_id in test_seg_ids:
        b = test_df[test_df["s_no"] == before_id]["meter_reading"][-window_size // 2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    if before_id in train_seg_ids:
        b = train_df[train_df["s_no"] == before_id]["meter_reading"][-window_size // 2:]
        segment = np.concatenate([b, segment])
        id_out["window_b_included"] = True

    # add window length from segment after at back (if available)
    after_id = id + 1
    if after_id in test_seg_ids:
        a = test_df[test_df["s_no"] == after_id]["meter_reading"][:window_size // 2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    if after_id in train_seg_ids:
        a = train_df[train_df["s_no"] == after_id]["meter_reading"][:window_size // 2]
        segment = np.concatenate([segment, a])
        id_out["window_a_included"] = True

    logger.info(f'diff in length :, {len(id_df) - len(segment)}, ({id_out["window_b_included"], id_out["window_a_included"]})')
    # each segment will have subsequences of overlapping windows:
    X = split_sequence(segment, window_size)
    id_out["X"] = X
    Anom_X = split_sequence(id_df["anomaly"], window_size)
    Isanom = Anom_X.sum(axis=1)
    id_out["labels"] = Isanom
    # reconstruct & errors
    criterion = torch.nn.MSELoss()
    vae.eval()  # batch norm in static mode
    X = torch.tensor(X, device=device).view(X.shape[0], 1, -1)
    Z, X_, loss = reconstruct(X, 75, vae, criterion, nz)
    id_out["Z"] = Z
    id_out["X_"] = X_
    id_out["recon_loss"] = loss

    test_out[id] = id_out


# Store the dict as pickle
with open(f'reconstruction_{b_id}.pkl', 'wb') as file:
    pickle.dump(test_out, file)

logger.info(f"reconstruction file created: reconstruction_{b_id}.pkl")