
print(__file__)
from utilz2 import *
################################################################################
##
weights_file=''
figure_file=''
stats_file=''
if 'project_' in __file__:
    import sys,os
    sys.path.insert(0,os.path.join(pname(pname(__file__)),'env'))
    weights_file=opj(pname(pname(__file__)),'net/weights',d2p(time_str(),'pth'))
    figure_file=opj(pname(pname(__file__)),'figures',d2p(time_str(),'pdf'))
    stats_file=opj(pname(pname(__file__)),'stats',d2p(time_str(),'txt'))
##
################################################################################
from projutils import *
from ..params.a import *
from .dataloader import *
from .stats import *
from ..net.code.net import *

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path='project_tac/18Jun24_14h09m36s_long_train',
)

d=2

dataiter = iter(trainloader)
for i in range(100):
    ms=[]
    oimages, labels = next(dataiter)
    print(oimages.size(),oimages.max(),oimages.min())
    sh(torchvision.utils.make_grid(oimages),1)
    #plt.savefig(figure_file)
    oimages=oimages.to(device)
    print(labels)
    for d in [4]:#[1,2,3,4,5]:
        for q in [0]:#[-1,0,1]:
            m=np.zeros((32,32))
            for x in range(32):
                for y in range(32):
                    images=1*oimages
                    #
                    images[:,1,max(x-d,0):min(x+d,32),max(y-d,0):min(y+d,32)]=q
                    outputs=net(images).detach().cpu().numpy()
                    m[x,y]+=outputs[0,labels.item()]
                #m[max(x-d,0):min(x+d,32),max(y-d,0):min(y+d,32),:]+=outputs[0,labels.item()]
            m=np.abs(m-m.flatten().mean())
            ms.append(m)
    m=na(ms).sum(axis=0)
    sh(m,2,r=0)
    m3=zeros((32,32,3))
    for i in range(3):
        m3[:,:,i]=z2o(m)

    cm()

print('*** Done')

#EOF