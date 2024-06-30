## 79 ########################################################################
from utilz2 import *
if host_name=='hiMac':
	run_path='project_tac/29Jun24_23h15m55s'
elif host_name=='jane':
	run_path='project_tac/25Jun24_19h30m07s'
else:
    run_path='project_tac/29Jun24_23h15m55s'
device='cuda:1'
batch_size=1
num_workers=4
datapath=opjD('data/rf_gen128_0')
repeats=10**6
