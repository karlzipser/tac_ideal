## 79 ########################################################################

print(__file__)
from utilz2 import *
##############################################################################
##
if 'project_' in __file__:
  import sys,os
  sys.path.insert(0,os.path.join(pname(pname(__file__)),'env'))
  figures_path=opj(pname(pname(__file__)),'figures')
##
##############################################################################
from projutils import *
from ..params.a import *
from .dataloader import *

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
  device=device,
  run_path=run_path,
)

dataiter = iter(trainloader)
for i in range(100):
  ms=[]
  oimages, labels = next(dataiter)
  print(oimages.size(),oimages.max(),oimages.min())
  
  plt.savefig(opj(figures_path,time_str()+'.pdf'))
  oimages=oimages.to(device)
  print(labels)
  for d in [1,2,3,4,5]:
    for q in [-1,0,1]:
      m=np.zeros((32,32))
      for x in range(32):
        for y in range(32):
            images=1*oimages
            images[:,1,max(x-d,0):min(x+d,32),max(y-d,0):min(y+d,32)]=q
            outputs=net(images).detach().cpu().numpy()
            m[x,y]+=outputs[0,labels.item()]
      m=np.abs(m-m.flatten().mean())
      ms.append(m)
  m=na(ms).sum(axis=0)
  CA()
  sh(m,2,r=0)
  sh(torchvision.utils.make_grid(oimages),1)
  m3=zeros((32,32,3))
  for i in range(3):
      m3[:,:,i]=z2o(m)

  #cm()

print('*** Done')

#EOF
## 79 ########################################################################
