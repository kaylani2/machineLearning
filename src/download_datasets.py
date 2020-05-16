## @TODO: Compare hashes
## @TODO: Fix unsw-nb15 formatting

import os
import sys

CICIDS = {
  'url' : '205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip',
  'dir' : 'datasets/cicids'
}

NSLKDD = {
  'url' : '205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip',
  'dir' : 'datasets/nslkdd'
}

UNSW_NB15 = {
  'url' : 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2F&files=UNSW-NB15%20-%20CSV%20Files&downloadStartSecret=4j3twms4roi',
  'dir' : 'datasets/unsw-nb15'
}

DATASETS = [CICIDS, NSLKDD, UNSW-NB15]

os.chdir ('../') #root
os.system ('mkdir datasets')
for dataset in DATASETS:
  os.system ('mkdir ' + dataset ['dir'])
  os.chdir (dataset ['dir'])
  os.system ('wget ' + dataset ['url'])
  os.system ('unzip *')
  os.chdir ('../') #datasets

os.chdir ('../') #root

sys.exit ()
