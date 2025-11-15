import os
import h5py
from jax import random


def get_3dshapes_dataset(rng, path):
    print("data is now loading...")
    shapes_data = h5py.File(path, 'r')
    data = shapes_data.get('images')[()]
    print("done!")
    ds = {}
    data = random.permutation(rng, data)
    ds['train'], ds['test'] = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
    return ds['train'], ds['test']