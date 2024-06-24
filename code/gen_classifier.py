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

stats_recorders={}
stats_keys=['train loss','test loss',]
net=get_net(device=device,net_class=Net)
for k in stats_keys:
    s=p.loss_s
    if 'test loss' in k:
        s*=p.test_sample_factor
    stats_recorders[k]=Loss_Recorder(
        stats_path,
        pct_to_show=p.percent_loss_to_show,
        s=s,
        name=k,
        )

"""
criterion=p.criterion
if p.opt==optim.Adam:
    optimizer = p.opt(net.parameters(),lr=p.lr)
else:
    optimizer = p.opt(net.parameters(),lr=p.lr,momentum=p.momentum)

save_timer=Timer(p.save_time)
test_timer=Timer(p.test_time)
max_timer=Timer(p.max_time)
"""
show_timer=Timer(p.show_time);show_timer.trigger()
loss_ctr=0
loss_ctr_all=0
it_list=[]
running_loss_list=[]

print('*** Start Training . . .')

def show_sample_outputs(outputs,labels):
    outputs=outputs.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    print(outputs)
    print(labels)
    print(shape(outputs))
    print(shape(labels))
    for i in range(16):
        o=outputs[i,:,0,0]
        l=0*o
        l[labels[i]]=1
        clf()
        plot(o/o.max(),'r')
        plot(o,'k')
        plot(l,'b')
        cm()

figure('train examples',figsize=(8,4))

    stats,acc_mean=get_accuracy(net,testloader,classes,device)
    print(time_str('Pretty2'),'epoch=',epoch)
    print(stats)
    t2f(opj(stats_path,time_str()+'.txt'),
        d2s(time_str('Pretty2'),'epoch=',epoch,'\n\n')+stats)
    ec=external_ctr=stats_recorders['train loss'].i[-1]
    stats_recorders['test accuracy'].do(
        acc_mean,
        external_ctr=ec,)
    kprint(stats_recorders['test accuracy'].__dict__)
    stats_recorders['test accuracy'].plot(fig='test accuracy',savefig=True)
    spause()




print('*** Done')

#EOF
## 79 ########################################################################
