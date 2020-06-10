import os
import sys
import wget
import ssl
import hashlib

CICIDS = {
  'url'  : 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip',
  'dir'  : 'datasets/cicids',
  'name' : 'MachineLearningCSV.zip',
  'hash' : 'c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928'
}

NSLKDD = {
  'url'  : 'http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip',
  'dir'  : 'datasets/nslkdd',
  'name' : 'NSL-KDD.zip',
  'hash' : 'b28a4ac1ab5b3f659d251d628f297630f9f0ac61403c4abf18776439553addb0'
}

UNSW_NB15 = {
  'url'  : 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2F&files=UNSW-NB15%20-%20CSV%20Files&downloadStartSecret=4j3twms4roi',
  'dir'  : 'datasets/unsw-nb15',
  'name' : 'UNSW-NB15%20-%20CSV%20Files.zip',
  'hash' : '4263563348ee50d3bfe327aa4dd360367f7c4b2f37e487838a579123cc068a83'
}

CTU_13 = { ### K: 1.9GB, contains .pcap files and botnets .exe
  'url'  : 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2',
  'dir'  : 'datasets/ctu-13',
  'name' : 'CTU-13-Dataset.tar.bz2',
  'hash' : '1f8daeca146131a368b432b2d8625de5e4429bce833f3e4cf4bea8312344ab7f'
}



DATASETS = [CICIDS, NSLKDD, UNSW_NB15, CTU_13]


os.chdir ('../') #root
os.system ('mkdir datasets')
for dataset in DATASETS:
  os.system ('mkdir ' + dataset ['dir'])
  os.chdir (dataset ['dir'])
  ssl._create_default_https_context = ssl._create_unverified_context
  wget.download (dataset ['url'], './')
  if (dataset ['name'].find ('.zip') != -1):
    os.system ('unzip ' + dataset ['name'])
  elif (dataset ['name'].find ('.tar.bz2') != -1):
    pass
    #os.system ('tar xjf ' + dataset ['name'] +  ' -v')

  ## Compare hashes
  BLOCK_SIZE = 65536
  fileHash = hashlib.sha256 ()
  with open (dataset ['name'], 'rb') as f:
    fileBlock = f.read (BLOCK_SIZE)
    while (len (fileBlock) > 0):
      fileHash.update (fileBlock)
      fileBlock = f.read (BLOCK_SIZE)
  if (fileHash.hexdigest () != dataset ['hash']):
    print ('File downloaded does not match!')

  os.chdir ('../../') #root

sys.exit ()
