
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

def ____cuda_to_rgb_image(cu):
    if len(cu.size()) == 3:
        return (255*(cu.detach().cpu().numpy().transpose(1,2,0))).astype(np.uint8)
    elif len(cu.size()) == 4:
        return z55(cu.detach().cpu().numpy()[0,:].transpose(1,2,0))
    else:
        assert False

dataiter = iter(trainloader)
for i in range(100):
    ms=[]
    oimages, labels = next(dataiter)
    print(oimages.size(),oimages.max(),oimages.min())
    #sh(torchvision.utils.make_grid(oimages),1)
    #plt.savefig(figure_file)
    oimages=oimages.to(device)
    print(labels)
    sh(cuda_to_rgb_image(oimages[0,:]),2)
    imgdic={}
    outdic={}
    vals=[10,12,14,16]
    for w in vals:
        for h in vals:
            m=np.zeros((32,32))
            for x in range(32):
                for y in range(32):
                    if x-w<0:
                        continue
                    if y-h<0:
                        continue                    
                    if x+w>32:
                        continue
                    if y+h>32:
                        continue
                    images=1*oimages
                    #print(images.min(),images.max())
                    if False:
                        zerosimg=0*images
                        zerosimg[:,:,max(x-w,0):min(x+w,32),max(y-h,0):min(y+h,32)]=1
                        randomized=1*images.view(-1)
                        randomized=randomized[torch.randperm(randomized.size()[0])]
                        randomized=randomized.view(images.size())
                        randomized*=(1-zerosimg)
                        #sh(cuda_to_rgb_image(randomized),6)
                        images*=zerosimg
                        images+=0.2*randomized
                    images=F.interpolate(images[:,:,x-w:x+w,y-h:y+h],size=(32,32),mode='bilinear',align_corners=False)
                    outputs=net(images).detach().cpu().numpy()
                    #m[x,y]+=outputs[0,labels.item()]
                    #images[0,:,0,0]=1
                    #images[0,:,0,1]=-1
                    #sh(cuda_to_rgb_image(images),1)
                    outdic[x,y,w,h]=outputs[0,labels.item()][0][0]
                    #cg(outdic[x,y,w,h])
                    imgdic[x,y,w,h]=1*images

                #m[max(x-d,0):min(x+d,32),max(y-d,0):min(y+d,32),:]+=outputs[0,labels.item()]
            #m=np.abs(m-m.flatten().mean())
            #ms.append(m)
    bestout=-1
    ov=[]
    for k in outdic:
        ov.append(outdic[k])
        #print(outdic[k],bestout,outdic[k]>bestout)
        if outdic[k]>bestout:
            bestout=outdic[k]
            bestxy=k
    #print(outdic)
    sh(imgdic[bestxy],4)
    x,y,w,h=bestxy
    sh(cuda_to_rgb_image(oimages[0,:]),10)
    figure(10)
    plot(na([y-h,y-h,y+h,y+h,y-h])-0.5,na([x-w,x+w,x+w,x-w,x-w])-0.5,'r')
    cm()

print('*** Done')

#EOF