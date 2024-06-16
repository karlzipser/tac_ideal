
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
    run_path='project_tac/15Jun24_20h34m46s-net_net',
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
layers=[1,2,3,4]
#layers=[1,2,3,]#4,5,6]
from utilz2.torch_ import *
from skimage import color

jitters=[-1,1]+8*[0]
for target_neuron in range(256):
    try:
        input_image = torch.randn(1, 3, 32, 32, requires_grad=True,device='cuda:0')
        input_image_big = torch.randn(1, 3, 32+4, 32+4, requires_grad=False,device='cuda:0')
        optimizer = optim.Adam([input_image], lr=0.01, weight_decay=1e-6)
        for i in range(500):
            input_image.requires_grad=False
            dx=np.random.choice(jitters)
            dy=np.random.choice(jitters)
            input_image_big[0,:,1+dx:1+32+dx,1+dy:1+32+dy]=input_image
            input_image[0,:,:,:]=input_image_big[0,:,1:32+1,1:32+1]
            img=cuda_to_rgb_image(input_image)
            img_hsv=color.rgb2hsv(img)
            avg_saturation=np.mean(img_hsv[:,:,1])
            #print(avg_saturation)
            #sh(img,9,r=1)
            input_image.requires_grad=True

            optimizer.zero_grad()
        
            x = input_image
            for j in layers:
                #print(j,x.size())
                #if j==4:
                #    x = torch.flatten(x,start_dim=1)
                    #print('\t',j,x.size())
                x = model[j](x)
            tmean=-x[0, target_neuron].mean()
            #print(tmean)

            imgmean=torch.abs(input_image-input_image.mean()).mean()
            #print(tmean,torch.abs(tmean)*imgmean)
            loss = tmean+3*torch.abs(tmean)*imgmean/(.1+avg_saturation)   #input_image.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                input_image.clamp_(0, 1)
            #if (i + 1) % 50 == 0:
            #    print(f'Iteration {i + 1}, Loss: {loss.item()}')

        optimized_image = input_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        for i in range(3):
            blank[:,:,i]=(255*z2o(optimized_image[:,:,i])).astype(np.uint8)
        n=layers[-1]
        if n not in imgs:
            imgs[n]={}
        imgs[n][str(target_neuron)]=1*blank
        figure(d2s(n),figsize=(9,9))
        sh(imgs[n],d2s(n),r=0,use_dict_keys_as_titles=False)#optimized_image,target_neuron,r=0)
#,b


    
    except KeyboardInterrupt:
        cr('*** KeyboardInterrupt ***')
        sys.exit()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('Exception!')
        print(d2s(exc_type,file_name,exc_tb.tb_lineno))   
    
savefigs()
input('here')


#EOF