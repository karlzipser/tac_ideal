## 79 ########################################################################

print(__file__)
from utilz2 import *
##############################################################################
##
if 'project_' in __file__:
    import sys,os
    sys.path.insert(0,os.path.join(pname(pname(__file__)),'env'))
    figures_path=opj(pname(pname(__file__)),'figures',)
##
##############################################################################
from projutils import *
from ..params.a import *

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path=run_path,
)

model=[]
for m in net.modules():
    model.append(m)

for i in rlen(model):
    print(i,model[i])

imgs={}

blank=get_blank_rgb(32,32)

#,a
imgs={}
blank=get_blank_rgb(32,32)
layers=[1] #4,5,6]
layers=[1]#4,5,6]
layers=[1,2,3,4,5]
#layers=[1,2,3,]#4,5,6]
from utilz2.torch_ import *
from skimage import color

from .dataloader import *
dataiter = iter(trainloader)
oimages, labels = next(dataiter)
oimages=oimages.to(device)
bkp=1*oimages
xsum=0*bkp
xctr=0
print(oimages.size())
sh(oimages,10,title=str(time.time()))
jitters=[-1,1]+10*[0]
for target_neuron in [labels.item()]:#range(10):
    #print(target_neuron)
    if True:#try
        for j in layers:
            model[j].requires_grad=False
        input_image = 1+1.*torch.randn(1, 3, 32, 32,
            requires_grad=False,device=device)
        #input_image2 = torch.randn(1, 3, 32, 32,
        #    requires_grad=False,device=device)
        input_image_big = torch.randn(1, 3, 32+4, 32+4,
            requires_grad=False,device=device)
        optimizer = optim.Adam([input_image], lr=.1)#, weight_decay=1e-6)
        for i in range(50000):
            oimages=1*bkp
            input_image.requires_grad=False
            if True:
                dx=np.random.choice(jitters)
                dy=np.random.choice(jitters)
                input_image_big[0,:,1+dx:1+32+dx,1+dy:1+32+dy]=input_image
                input_image[0,:,:,:]=input_image_big[0,:,1:32+1,1:32+1]
            input_image*=0.9
            input_image+=0.1*torch.randn(1, 3, 32, 32,device=device)
                #img=cuda_to_rgb_image(input_image)
                #img_hsv=color.rgb2hsv(img)
                #avg_saturation=np.mean(img_hsv[:,:,1])
            for q in [0,2]:
                input_image[0,q,:,:]=input_image[0,1,:,:]

            optimizer.zero_grad()
        
            #x = input_image*oimages
            with torch.no_grad():
                input_image.clamp_(0.25, 1)
            oimages.requires_grad=True
            input_image.requires_grad=True
            #x=1*oimages
            #for q in range(3):
            #    x[0,q,:,:]=x[0,q,:,:]*input_image
            x=input_image*oimages
            xsum+=x
            xctr+=1
            xx=1*x
            for j in layers:
                x = model[j](x)
            print(x)
            #oimages.requires_grad=True
            #if target_neuron>=len(x):
            #    continue
            #print('*****',x.size(),target_neuron)
            #cg(x.max().item())
            tmean=-x[0, target_neuron].mean()
            #imgmean=torch.abs(input_image-input_image.mean()).mean()
            loss = tmean#-x.max()  #tmean#+3*torch.abs(tmean)#*imgmean/(.1+avg_saturation)
            loss.backward()
            optimizer.step()

            #sh(xx,1,r=0,use_dict_keys_as_titles=False,title=str(time.time()))
            sh(input_image,4,title=classes[labels.item()],r=0,use_dict_keys_as_titles=False)
            sh(xsum/xctr,5,title=classes[labels.item()],r=0,use_dict_keys_as_titles=False)
        optimized_image = (1*oimages*input_image).detach().cpu().numpy()[0].transpose(1, 2, 0)
        for i in range(3):
            blank[:,:,i]=(255*z2o(optimized_image[:,:,i])).astype(np.uint8)
        n=layers[-1]
        if n not in imgs:
            imgs[n]={}
        imgs[n][str(target_neuron)]=1*blank
        figure(d2s(n),figsize=(9,9))
        sh(imgs[n],title(classes(labels.item())),r=0)

    """
    except KeyboardInterrupt:
        cr('*** KeyboardInterrupt ***')
        sys.exit()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('Exception!')
        print(d2s(exc_type,file_name,exc_tb.tb_lineno))   
    """
savefigs()
print('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
input('here')


#EOF