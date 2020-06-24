"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import sys
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

from random import shuffle

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"YOUR DIRECTORY HERE"
        self.n_epochs = 8
        self.learning_rate = 0.0002
        self.batch_size = 32
        self.patch_size = 64
        self.test_results_dir = "RESULTS GO HERE"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    c.root_dir = '../out'
    c.test_results_dir = './Results-dir'

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = list(range(len(data)))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    
    shuffle(keys)
    
    train_size = int(len(keys) * .8)
    train_keys = keys[:train_size]
    
    non_train_keys = keys[train_size:]
    valid_keys = non_train_keys[:len(non_train_keys)//2]
    test_keys = non_train_keys[len(non_train_keys)//2:]

    print(f'Random split. Train: {len(train_keys)}, valid: {len(valid_keys)}, test: {len(test_keys)}')
    assert np.intersect1d(train_keys, valid_keys).tolist() == [], 'Intersection train & valid not empty'
    assert np.intersect1d(train_keys, test_keys).tolist() == [], 'Intersection train & test not empty'
    assert np.intersect1d(valid_keys, test_keys).tolist() == [], 'Intersection valid & test not empty'
    
    sorted_union = np.sort(np.union1d(np.union1d(train_keys, valid_keys), test_keys))
    assert sorted_union.tolist() == np.arange(len(data)).tolist(), 'Union does not contain all keys'

    split['train'] = train_keys
    split['val'] = valid_keys
    split['test'] = test_keys
    
    
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    json_filename = os.path.join(exp.out_dir, "results.json")
    with open(json_filename, 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
    print(f'Done. Results written to: {json_filename}')

