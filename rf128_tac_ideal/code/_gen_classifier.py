## 79 ########################################################################
# branch master
print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
from projutils import *
from .dataloader import *
from ..net.code.net import *

thispath=pname(pname(__file__))
sys.path.insert(0,opj(thispath,'env'))
weights_path=opj(thispath,'net/weights')
figures_path=opj(thispath,'figures')
stats_path=opj(thispath,'stats')

mkdirp(figures_path)
mkdirp(stats_path)

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path=run_path,
)




stats,acc_mean=get_accuracy(net,testloader,classes,device)
print(stats)





print('*** Done')

#EOF
## 79 ########################################################################
