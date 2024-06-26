#file: 'tac_ideal/code/findideal.py'
## 79 ########################################################################
# python projutils/project.py --src tac_ideal --termout 0
# python projutils/view.py --src project_tac_ideal
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
import torch
#import torch.nn as nn
#import torch.nn.parallel
#import torch.optim as optim
#import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as v2


def get_transforms(d,image_size):

    geometric_transforms_list=[]

    k='RandomPerspective'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomPerspective(
                distortion_scale=d['RandomPerspective_distortion_scale'],
                p=d['RandomPerspective_p'],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=d['RandomPerspective_fill'],
            )
        )

    k='RandomRotation'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomRotation(d['RandomRotation_angle'],fill=d['RandomRotation_fill'])
        )

    k='RandomZoomOut'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomZoomOut(side_range=d['RandomZoomOut_side_range'],fill=d['RandomZoomOut_fill'])
        )


    k='Pad'
    if k in d and d[k]:
        geometric_transforms_list.append(
            #v2.Pad(padding=(640-360)//2,fill=d['Pad_fill)
            v2.Pad(padding=64,fill=d['Pad_fill'])
        )

    k='CenterCrop'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.CenterCrop(size=640)
        )

    
    k='RandomResizedCrop'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomResizedCrop(
                image_size,
                scale=d['RandomResizedCrop_scale'],
                ratio=d['RandomResizedCrop_ratio'],
                antialias=True,
            )
        )

    k='Resize'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.Resize(size=image_size,antialias=True)
        )
    
    k='RandomHorizontalFlip'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomHorizontalFlip(p=d['RandomHorizontalFlip_p'])
        )
        
    k='RandomVerticalFlip'
    if k in d and d[k]:
        geometric_transforms_list.append(
            v2.RandomVerticalFlip(p=d['RandomVerticalFlip_p'])
        )

    color_transforms_list=[]
    k='ColorJitter'
    if k in d and d[k]:
        color_transforms_list.append(
            v2.ColorJitter(
                brightness=d['ColorJitter_brightness'],
                contrast=d['ColorJitter_contrast'],
                saturation=d['ColorJitter_saturation'],
                hue=d['ColorJitter_hue'],
            )
        )

    return geometric_transforms_list,color_transforms_list


_fill=(0,0,0)
transforms_dict=dict(
    RandomPerspective=True,
    RandomPerspective_distortion_scale=0.3,
    RandomPerspective_p=0.3,
    RandomPerspective_fill=_fill,

    RandomRotation=True,
    RandomRotation_angle=12,
    RandomRotation_fill=_fill,

    RandomResizedCrop=True,
    RandomResizedCrop_scale=(0.85,1),
    RandomResizedCrop_ratio=(0.85,1.2),

    RandomHorizontalFlip=True,
    RandomHorizontalFlip_p=0.5,
        
    RandomVerticalFlip=False,
    RandomVerticalFlip_p=0.5,

    RandomZoomOut=True,
    RandomZoomOut_fill=_fill,
    RandomZoomOut_side_range=(1.0,1.5),

    ColorJitter=False,
    ColorJitter_brightness=(0,1),
    ColorJitter_contrast=(0,1),
    ColorJitter_saturation=(0,2),
    ColorJitter_hue=(-.03,.03),
)
transforms_dict2=dict(
    RandomPerspective=True,
    RandomPerspective_distortion_scale=0.5,
    RandomPerspective_p=0.5,
    RandomPerspective_fill=_fill,

    RandomRotation=True,
    RandomRotation_angle=16,
    RandomRotation_fill=_fill,

    RandomResizedCrop=True,
    RandomResizedCrop_scale=(0.75,1),
    RandomResizedCrop_ratio=(0.75,1.2),

    RandomHorizontalFlip=True,
    RandomHorizontalFlip_p=0.5,
        
    RandomVerticalFlip=False,
    RandomVerticalFlip_p=0.5,

    RandomZoomOut=True,
    RandomZoomOut_fill=_fill,
    RandomZoomOut_side_range=(1.0,1.5),

    ColorJitter=False,
    ColorJitter_brightness=(0,1),
    ColorJitter_contrast=(0,1),
    ColorJitter_saturation=(0,2),
    ColorJitter_hue=(-.03,.03),
)
geometric_transforms_list,color_transforms_list=get_transforms(
    d=transforms_dict,
    image_size=(32,32))


for repeat in range(repeats):
    cE(repeat)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    net=get_net(
        device=device,
        run_path=run_path,
        latest=True,
    )

    model=nn.ModuleList()
    for m in net.modules():
        model.append(m)

    for i in rlen(model):
        print(i,model[i])

    imgs={}
    save_timer=Timer(1)
    blank=get_blank_rgb(32,32)

    def show_sample_outputs(outputs,labels):
        outputs=outputs.detach().cpu().numpy()
        for i in range(1):
            o=outputs[i,:,0,0]
            l=0*o
            l[labels[i]]=1
            clf()
            plot(o/o.max(),'r')
            #plot(o,'k')
            plot(l,'b')
            if labels[i]==np.argmax(outputs):
                answer=True
            else:
                answer=False
            return answer
            #cm()
    #
    imgs={}
    blank=get_blank_rgb(32,32)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def custom_clip_grads(parameters, clip_value):
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data.clamp(min=-clip_value, max=clip_value)

    def save_img(optimized_image,category,path):
        #blank=get_blank_rgb(32,32)
        #blank=255*z2o(optimized_image).astype(np.uint8)
        mkdirp(opj(path,category))
        imsave(opj(path,category,time_str()+'.png'),fix_bgr(cuda_to_rgb_image(optimized_image)))








    from torch.utils.data import DataLoader, Dataset
    class GenDataset(Dataset):
        def __init__(self, root, transform=None):
            cy('GenDataset __init__()')
            self.root = root
            self.transform = transform
            self.images = []
            self.labels = []
            fs0=sggo(opjD('data/gen0/*'))
            for cf in fs0:
                if not os.path.isdir(cf):
                    continue
                for image in sggo(cf,'*.png'):
                    self.images.append(image)
                    self.labels.append(fname(cf))
                    #print(self.images[-1],self.labels[-1])
            print('len(self.images)=',
                len(self.images),'len(self.labels)=',len(self.labels))
        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image = rimread(self.images[index])
            #sh(z55(image),title=d2s(image.max(),image.min()),r=0)
            if self.transform:
                image = self.transform(image)
                #image-=0.5
                #image*=2.
            return image, self.labels[index]

    classes2nums={}
    for iii in rlen(classes):
        classes2nums[classes[iii]]=iii














    from utilz2.torch_ import *
    from skimage import color
    for layers in [
        [1,2,3,4,5],
    ]:
        target_neuron=0
        ntimer=Timer(5*minutes)
        jitters=[-1,1]+100*[0]
        todo=list(range(10))
        while todo:
            target_neuron=todo.pop(0)
            if True:#try:


                gen_train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]+geometric_transforms_list)
                gen_traindata = GenDataset(
                    root=opjD('data/gen0'), transform=gen_train_transform)
                gen_trainloader = DataLoader(
                    gen_traindata, batch_size=batch_size, shuffle=True)
                gen_train_dataiter=iter(gen_trainloader)

                while True:
                    train_inputs,train_labels=next(gen_train_dataiter)
                    print(classes2nums[train_labels[0]],target_neuron)
                    if train_labels[1]==target_neuron:
                        break
                input_image=train_inputs
                cb(input_image.size())
                #input_image = torch.randn(1, 3, 32, 32,
                input_image=input_image.to(device)
                input_image.requires_grad=True

                input_image_big = torch.randn(1, 3, 32+4, 32+4,
                    requires_grad=False,device=device)
                optimizer = optim.Adam([input_image], lr=0.5, weight_decay=1e-6)
                for i in range(5000000):
                    if ntimer.rcheck():
                        break
                    if len(sggo(datapath,classes[target_neuron],'*.png'))>=10000000:
                        break
                    input_image.requires_grad=False
                    dx=np.random.choice(jitters)
                    dy=np.random.choice(jitters)
                    input_image_big[0,:,1+dx:1+32+dx,1+dy:1+32+dy]=input_image
                    input_image[0,:,:,:]=input_image_big[0,:,1:32+1,1:32+1]
                    try:
                        img=cuda_to_rgb_image(input_image)
                        img_hsv=color.rgb2hsv(img)
                        avg_saturation=np.mean(img_hsv[:,:,1])
                        input_image.requires_grad=True
                    except:
                        print('exception: img=cuda_to_rgb_image(input_image)')
                        input_image=1*input_image_prev
                        save_img(input_image,classes[target_neuron],datapath)
                        #input_image.requires_grad=True
                        if len(sggo(datapath,classes[target_neuron],'*.png'))<10000000:
                            todo.append(target_neuron)
                        break
                        #ntimer.reset()
                        #continue
                    optimizer.zero_grad()
                
                    x = input_image
                    ctr=1
                    for j in layers:
                        x = model[j](x)
                        if ctr<5:
                            x=torch.nn.functional.relu(x)
                        ctr+=1
                    with torch.no_grad():
                        input_image_prev=1*input_image
                    tmean=-x[0, target_neuron].mean()
                    y=3*x
                    y[0,target_neuron]=0
                    y=y/y.max()
                    ymean=y[0, target_neuron].mean()
                    imgmean=torch.abs(input_image-input_image.mean()).mean()
                    #print(tmean,imgmean,avg_saturation)
                    loss = 50*y.max()+ymean+tmean+1.*torch.abs(tmean)*(imgmean+avg_saturation)
                    loss.backward()
                    clip_value = 0.01
                    custom_clip_grads(model.parameters(), clip_value)
                    optimizer.step()
                    with torch.no_grad():
                        input_image.clamp_(-1, 1)
                    if not i%100:
                        figure(10,figsize=(3,3))
                        sh(input_image,10)
                        figure(11,figsize=(3,3))
                        answer=show_sample_outputs(x,[target_neuron])
                        plt.title(d2s(classes[target_neuron],answer))
                        spause()
                        if save_timer.rcheck() and ntimer.time()>10:
                            if answer:
                                save_img(input_image,classes[target_neuron],opjD('data/gen1'))
                            else:
                                save_timer.trigger()
                optimized_image = input_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
                for i in range(3):
                    blank[:,:,i]=(255*z2o(optimized_image[:,:,i])).astype(np.uint8)
                n=layers[-1]
                if n not in imgs:
                    imgs[n]={}
                imgs[n][str(target_neuron)]=1*blank
                figure(d2s(n),figsize=(9,9))
                sh(imgs[n],d2s(n),r=0,use_dict_keys_as_titles=False)
            """
            except KeyboardInterrupt:
                cr('*** KeyboardInterrupt ***')
                sys.exit()
            except Exception as e:
                pass
                #exc_type, exc_obj, exc_tb = sys.exc_info()
                #file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                #print('Exception!')
                #print(d2s(exc_type,file_name,exc_tb.tb_lineno))   
            """

    savefigs(figures_path)
    print('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #input('here')


#EOF
