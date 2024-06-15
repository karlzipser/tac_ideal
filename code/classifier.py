
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
    run_path='project_tac/15Jun24_13h42m54s-jake_long_train',
)

model = [module for module in net.modules() if not isinstance(module, nn.Sequential)]

for i in range(100):
    print(i,model[i])

imgs={}

blank=get_blank_rgb(128,128)

#,a
layers=[2]
layers=[2,4,5]
layers=[2,4,5,8,9]
layers=[2,4,5,8,9,12,13]
layers=[2,4,5,8,9,12,13,16,17]
#layers=[2,4,5,8,9,12,13,16,17,20,21]
for target_neuron in range(64):
    if True:#try:
        input_image = torch.randn(1, 1, 128, 128, requires_grad=True,device='cuda:0')
        input_image_big = torch.randn(1, 1, 128+4, 128+4, requires_grad=False,device='cuda:0')
        optimizer = optim.Adam([input_image], lr=0.1, weight_decay=1e-6)
        for i in range(200):
            input_image.requires_grad=False
            dx=np.random.choice([-1,0,1])
            dy=np.random.choice([-1,0,1])
            input_image_big[0,0,1+dx:1+128+dx,1+dy:1+128+dy]=input_image
            input_image[0,0,:,:]=input_image_big[0,0,1:128+1,1:128+1]
            input_image.requires_grad=True

            optimizer.zero_grad()
        
            x = input_image
            for j in layers:
                x = model[j](x)
            loss = -x[0, target_neuron].mean()+0e7*input_image.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                input_image.clamp_(0, 1)
            if (i + 1) % 50 == 0:
                print(f'Iteration {i + 1}, Loss: {loss.item()}')

        optimized_image = input_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        for i in range(3):
            blank[:,:,i]=(255*z2o(optimized_image[:,:,0])).astype(np.uint8)
        n=layers[-1]
        if n not in imgs:
            imgs[n]={}
        imgs[n][str(target_neuron)]=1*blank
        figure(d2s(n),figsize=(18,18))
        sh(imgs[n],d2s(n),r=0,use_dict_keys_as_titles=False)#optimized_image,target_neuron,r=0)

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



#EOF