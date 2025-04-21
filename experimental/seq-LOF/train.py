import json
import pickle
from pyod.models.lof import LOF
import numpy as np

if __name__ == "__main__":
    # Load config
    with open('../../config.json', 'r') as file:
        config = json.load(file)

    prefix = "../../"

    # Training configs
    window_size = config['preprocessing']['window_size']
    signal_shape = window_size  # window-size
    n_epochs = config['training']['num_epochs']

    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)


    # Add dataset code here...
    X_train = np.load(f"X_train_{b_id}.npy")
    clf = LOF()
    clf.fit(X_train)
    with open(f"LOF_{b_id}","wb") as f:
      pickle.dump(clf,f)



