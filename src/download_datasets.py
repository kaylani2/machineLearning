import os
import sys

CICIDS_URL = '205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip'
NSLKDD_URL = '205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip'
urls = [CICIDS_URL, NSLKDD_URL]



os.chdir ('../') #root
os.system ('mkdir datasets datasets/cicids datasets/nslkdd')

for url in urls:
  os.system ('wget ' + url)

os.system ('mv MachineLearningCSV.zip datasets/cicids/')
os.system ('mv NSL-KDD.zip datasets/nslkdd/')
os.chdir ('datasets/cicids/')
os.system ('unzip MachineLearningCSV.zip')
os.chdir ('../nslkdd/')
os.system ('unzip NSL-KDD.zip')
os.chdir ('../../') #root
sys.exit ()
