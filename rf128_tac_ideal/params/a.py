## 79 ########################################################################
from utilz2 import *
run_path=most_recent_file_in_folder('project_tac')
device='cuda:1'
batch_size=1
num_workers=4
datapath=opjD('data/rf_gen128_0')
repeats=10**6
#lr=0.03 #good value
lr=0.015
