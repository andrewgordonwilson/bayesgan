#!/usr/bin/env python
import shutil
import os
import numpy as np
import glob

if __name__ == "__main__":

    data_path = "/data"
    dest_data_path = os.path.join(os.getenv("HOME"), "data")
    num_classes = 50
    
    dirnames = [rd.split("/")[-1] for rd in glob.glob(os.path.join(data_path, "imagenet/train_256/n*"))]

    rand_subset = np.random.choice(dirnames, num_classes, replace=False)

    for rdir in rand_subset:
        for train_or_val in ["train", "val"]:
            source_dir = os.path.join(data_path, "imagenet/%s_256/%s" % (train_or_val, rdir))
            dest_dir = os.path.join(dest_data_path, "imagenet_mini/%s_256/%s" % (train_or_val, rdir))
            print "Copying %s to %s" % (source_dir, dest_dir)
            shutil.copytree(source_dir, dest_dir)

