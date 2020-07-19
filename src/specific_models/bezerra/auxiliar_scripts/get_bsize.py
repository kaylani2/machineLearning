import os

DATASET_DIR = '../../../datasets/Dataset-IoT/'
NETFLOW_DIRS = ['MC/NetFlow/', 'SC/NetFlow/', 'ST/NetFlow/']


# MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets'
# MC_I_SECOND: Has infected data from Mirai botnets
# MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
# MC_L: Has legitimate data, no infection


path_types = ['MC', 'SC', 'ST']
data_set_files = [ [r'MC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'SC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'ST_I{}.csv'.format(index) for index in range(1, 4)] ]

for path, files in zip(path_types, data_set_files):
    files.append(path + '_L.csv')

################
##reading data##
################
total_size = 0
for n, (path, files) in enumerate(zip(NETFLOW_DIRS, data_set_files), start=1):
    for csvFile in files:
        total_size += os.path.getsize(DATASET_DIR + path + csvFile)

total_size *= 1e-9
print ("Total size = ", total_size, "GB")
#Total size =  0.616358906 GB
