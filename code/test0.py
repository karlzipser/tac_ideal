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
        for category in os.listdir(root):
            for image in os.listdir(os.path.join(root, category)):
                self.images.append(os.path.join(root, category, image))
                self.labels.append(int(category))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = rimread(self.images[index])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

train_data = ImageDataset(root=opjD('data/gen0'), transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
iter=iter(train_loader)
for i in range(4):
    image,label=next(iter)
    cb(image)
    cy(label)
    sh(image,title=str(label[0]))
    time_sleep(1)

stats,acc_mean=get_accuracy(net,train_loader,list(range(10)),device)
print(stats)

#EOF
