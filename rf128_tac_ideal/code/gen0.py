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

for repeat in range(repeats):
    cE(repeat)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    net=get_net(
        device=device,
        run_path=run_path,
        latest=True,
    )
    #cb(net,r=1)
    model=nn.ModuleList()
    ctr=0
    for m in net.modules():
        model.append(m)
        #print(ctr,m)
        #print('_________________')
        ctr+=1

    for i in rlen(model):
        print(i,model[i])

    imgs={}
    save_timer=Timer(1)
    blank=get_blank_rgb(128,128)

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
            #cm()
    #
    imgs={}
    blank=get_blank_rgb(128,128)

    classes = dict(
        plane=0,
        car=1,
        bird=2,
        cat=3,
        deer=4,
        dog=5,
        frog=6,
        horse=7,
        ship=8,
        truck=9,
        )
    classes=kys(classes)

    def custom_clip_grads(parameters, clip_value):
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data.clamp(min=-clip_value, max=clip_value)

    def save_img(optimized_image,category,path):
        #blank=get_blank_rgb(128,128)
        #blank=255*z2o(optimized_image).astype(np.uint8)
        mkdirp(opj(path,category))
        imsave(opj(path,category,time_str()+'.png'),fix_bgr(cuda_to_rgb_image(optimized_image)))

    from utilz2.torch_ import *
    from skimage import color
    for layers in [
         list(range(3,25)),
    ]:
        target_neuron=0
        ntimer=Timer(10)
        jitters=[-1,1]+1000*[0]
        todo=list(range(10))
        while todo:
            target_neuron=todo.pop(0)
            if True:#try:
                input_image = torch.randn(1, 3, 128, 128,
                    requires_grad=True,device=device)
                input_image_big = torch.randn(1, 3, 128+4, 128+4,
                    requires_grad=False,device=device)
                optimizer = optim.Adam([input_image], lr=.5, weight_decay=1e-6)
                for i in range(5000000):
                    if ntimer.rcheck():
                        break
                    if len(sggo(datapath,classes[target_neuron],'*.png'))>=10000000:
                        break
                    input_image.requires_grad=False
                    dx=np.random.choice(jitters)
                    dy=np.random.choice(jitters)
                    input_image_big[0,:,1+dx:1+128+dx,1+dy:1+128+dy]=input_image
                    input_image[0,:,:,:]=input_image_big[0,:,1:128+1,1:128+1]
                    #print(input_image.min(),input_image.max())
                    if torch.isnan(input_image).sum().item()>0:
                        #cb('here',r=1)
                        input_image=1*input_image_prev+0.1*torch.randn(1, 3, 128, 128,
                            requires_grad=True,device=device)
                    try:
                        img=cuda_to_rgb_image(input_image)
                        img_hsv=color.rgb2hsv(img)
                        avg_saturation=np.mean(img_hsv[:,:,1])
                        input_image.requires_grad=True
                    except:
                        cr('exception: img=cuda_to_rgb_image(input_image)',r=0)
                        input_image=1*input_image_prev+0.1*torch.randn(1, 3, 128, 128,device=device)
                        continue
                        save_img(input_image,classes[target_neuron],datapath)
                        #input_image.requires_grad=True
                        if len(sggo(datapath,classes[target_neuron],'*.png'))<10000000:
                            todo.append(target_neuron)

                        break
                        #ntimer.reset()
                        #continue
                    optimizer.zero_grad()
                
                    x = input_image
                    
                    for j in layers:
                        #cm(x.size())
                        #cg(j,model[j])
                        #cb('-------------')
                        x = model[j](x)

                    with torch.no_grad():
                        input_image_prev=1*input_image
                    tmean=-x[0, target_neuron].mean()
                    y=3*x
                    y[0,target_neuron]=0
                    y=y/y.max()
                    ymean=y[0, target_neuron].mean()
                    imgmean=torch.abs(input_image-input_image.mean()).mean()
                    #print(tmean,imgmean,avg_saturation)
                    if True:
                        loss = 50*y.max()+ymean+tmean+1.*torch.abs(tmean)*(imgmean+avg_saturation)
                    else:
                        loss = 1*y.max()+ymean+tmean+1.*torch.abs(tmean)*(imgmean+0*avg_saturation)
                    loss.backward()
                    clip_value = 0.1
                    custom_clip_grads(model.parameters(), clip_value)
                    optimizer.step()
                    if not i%100:


                        figure(10,figsize=(3,3))
                        try:
                            sh(input_image,10)
                        except:
                            cE('sh failed')
                        figure(11,figsize=(3,3))
                        answer=show_sample_outputs(x,[target_neuron])
                        plt.title(d2s(classes[target_neuron],answer))
                        spause()
                        if save_timer.rcheck() and ntimer.time()>3:
                            if answer:
                                save_img(input_image,classes[target_neuron],datapath)
                            else:
                                save_timer.trigger()


                        """
                        figure(10,figsize=(3,3))
                        sh(input_image,10)
                        figure(11,figsize=(3,3))
                        show_sample_outputs(x,[target_neuron])
                        plt.title(classes[target_neuron])
                        spause()
                        """
                    with torch.no_grad():
                        input_image.clamp_(-1, 1)
                    if save_timer.rcheck() and ntimer.time()>3:
                        save_img(input_image,classes[target_neuron],datapath)
                optimized_image = input_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
                for i in range(3):
                    blank[:,:,i]=(255*z2o(optimized_image[:,:,i])).astype(np.uint8)
                n=layers[-1]
                if False:
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
