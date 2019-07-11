import random
import os

PRETRAIN_DIR = '/home/jzda/storage/ice/pretrain/features/pretrain'
TRAIN_DIR = '/home/jzda/storage/ice/pretrain/features/pretrain/train'
DEV_DIR = '/home/jzda/storage/ice/pretrain/features/pretrain/dev'
for filename in os.listdir(PRETRAIN_DIR):
    if filename.endswith('json'):
        if random.random() < 0.05:
            os.rename(os.path.join(PRETRAIN_DIR, filename), os.path.join(DEV_DIR, filename))
        else:
            os.rename(os.path.join(PRETRAIN_DIR, filename), os.path.join(TRAIN_DIR, filename))
