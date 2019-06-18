import numpy as np

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1).astype(int)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets


