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

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
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
                print(self.images[-1],self.labels[-1])
        cE(self.labels)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = rimread(self.images[index])
        image=fix_bgr(image)

        sh(z55(image),title=d2s(image.max(),image.min()),r=1)
        if self.transform:
            image = self.transform(image)
            image=image/255.
            image-=0.5
            image*=2.
        return image, self.labels[index]

train_data = ImageDataset(root=opjD('data/gen0'), transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                    shuffle=True, num_workers=num_workers)


def get_accuracy(net,testloader,classes,device):
    label_i={}
    for i in rlen(classes):
        label_i[classes[i]]=i
    correct_pred = {str(classname): 0 for classname in classes}
    total_pred = {str(classname): 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            #cb(images)
            #cy(labels)
            images=images.to(device)
            #labels=labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label_i[label] == prediction:
                    correct_pred[classes[label_i[label]]] += 1
                total_pred[classes[label_i[label]]] += 1
    stats=[]
    ctr=0
    acc_mean=0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        acc_mean+=accuracy
        ctr+=1
        stats.append(f'**** Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    acc_mean/=ctr
    stats.append(d2n('\tMean accuracy is ',int(acc_mean),'%.'))
    stats='\n'.join(stats)
    return stats,acc_mean


iter=iter(train_loader)
for i in range(4):
    image,label=next(iter)
    #cb(image)
    #cy(label)
    sh(image,title=str(label[0]))
    time_sleep(.1)
#classes=[0,1,2,3,4,5,6,7,8,9]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


stats,acc_mean=get_accuracy(net,train_loader,classes,device)
print(stats)

#EOF
