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

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path=run_path,
)

model=nn.ModuleList()
for m in net.modules():
    model.append(m)

for i in rlen(model):
    print(i,model[i])

imgs={}

blank=get_blank_rgb(32,32)

def show_sample_outputs(outputs,labels):
    outputs=outputs.detach().cpu().numpy()
    #labels=labels.detach().cpu().numpy()
    #print(outputs)
    #print(labels)
    #print(shape(outputs))
    #print(shape(labels))
    for i in range(1):
        o=outputs[i,:,0,0]
        l=0*o
        l[labels[i]]=1
        clf()
        plot(o/o.max(),'r')
        #plot(o,'k')
        plot(l,'b')
        #cm()
#,a
imgs={}
blank=get_blank_rgb(32,32)
layers=[1] #4,5,6]
layers=[1]#4,5,6]
layers=[1,2,3,4,5]
#layers=[1,2,3,]#4,5,6]
#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def custom_clip_grads(parameters, clip_value):
    for p in parameters:
        if p.grad is not None:
            p.grad.data = p.grad.data.clamp(min=-clip_value, max=clip_value)



from utilz2.torch_ import *
from skimage import color
for layers in [
    #[1],
    #[1,2],
    #[1,2,3],
    #[1,2,3,4],
    [1,2,3,4,5],
]:
    target_neuron=0
    ntimer=Timer(60)
    jitters=[-1,1]+100*[0]
    todo=list(range(10))
    #todo=[3]
    while todo:
        target_neuron=todo.pop(0)
    #for target_neuron in [2]:
        #print(target_neuron)
        if True:#try:
            input_image = torch.randn(1, 3, 32, 32,
                requires_grad=True,device=device)
            input_image_big = torch.randn(1, 3, 32+4, 32+4,
                requires_grad=False,device=device)
            optimizer = optim.Adam([input_image], lr=0.1, weight_decay=1e-6)
            for i in range(5000000):
                if ntimer.rcheck():
                    #outs[target_neuron]=
                    break
                input_image.requires_grad=False
                dx=np.random.choice(jitters)
                dy=np.random.choice(jitters)
                input_image_big[0,:,1+dx:1+32+dx,1+dy:1+32+dy]=input_image
                input_image[0,:,:,:]=input_image_big[0,:,1:32+1,1:32+1]
                #input_image=torch.nan_to_num(input_image)
                try:
                    img=cuda_to_rgb_image(input_image)
                    img_hsv=color.rgb2hsv(img)
                    avg_saturation=np.mean(img_hsv[:,:,1])
                    input_image.requires_grad=True
                except:
                    print('exception: img=cuda_to_rgb_image(input_image)')
                    input_image=1*input_image_prev
                    input_image.requires_grad=True
                    break
                    #sh(input_image,100,title=str(time_str('Pretty2'),r=0))      
                    #todo.append(target_neuron)
                    ntimer.reset()
                    continue


                optimizer.zero_grad()
            
                x = input_image
                ctr=1
                for j in layers:
                    x = model[j](x)
                    if ctr<5:
                        x=torch.nn.functional.relu(x)
                    ctr+=1

                #if target_neuron>=len(x):
                #    continue
                #print('*****',x.size(),target_neuron)
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
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

                optimizer.step()
                if not i%100:
                    figure(10,figsize=(3,3))
                    sh(input_image,10)
                    figure(11,figsize=(3,3))
                    show_sample_outputs(x,[target_neuron])
                    spause()
                with torch.no_grad():
                    input_image.clamp_(-1, 1)
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
input('here')


#EOF
