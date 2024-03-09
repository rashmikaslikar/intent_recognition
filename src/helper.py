import numpy as np
import pickle
from pathlib import Path

def save_data(data, path, name):
    """Save data to path"""
    path = f"{path}/{name}"
    Path(path).parent.mkdir(exist_ok=True)
    if Path(path).suffix == '.pkl':
        with open(path, 'wb') as f:  
            pickle.dump(data, f)
    elif Path(path).suffix=='.npy':
        np.save(path,data)

def load_data(path: str):
    """Load data from path"""
    file_path = Path(path)
    print(file_path)
    if file_path.suffix == ".npy":
        data = np.load(file_path)
    elif file_path.suffix == ".pkl":
        data = pickle.load(open(file_path, 'rb'))
    else:
        raise ValueError(
            "File format not supported. Please use a CSV or PKL file."
        )

    return data
        