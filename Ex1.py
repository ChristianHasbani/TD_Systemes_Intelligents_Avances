import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

# Learning Parameters (Constants)
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 20

# The data, split between train and test sets
train_dt = datasets.MNIST(
    root='./data',train=True,
    download= True, 
    transform=transforms.ToTensor()
    )

test_dt = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )

gen_train = DataLoader(
    dataset=train_dt, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0
    )

for interation, batch in enumerate(gen_train):
    print(interation, type(batch))
    print(len(batch))
    print(batch[0].shape,batch[1].shape)

# print(train_dt)
# print(test_dt)