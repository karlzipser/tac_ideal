## 79 ########################################################################
from utilz2 import *
if host_name=='hiMac':
	run_path='project_tac/20Jun24_12h29m11s'
	run_path='project_tac/21Jun24_23h10m13s'
elif host_name=='jane':
	run_path='project_tac/25Jun24_19h30m07s'
elif host_name=='jake':
	run_path='project_tac/29Jun24_20h29m40s'
device='cuda:1'
batch_size=1
num_workers=4
datapath=opjD('data/gen0')
repeats=10**6