# Fresh/Rotten Fruit Identifier

This is an implementation of the "Fundamentals of Deep Learning" Course by Nvidia.

I trained this model to be able to recognize fresh and rotten fruit, I was asked to validate the model to 92% and successfully was able to reach 94%, which exceeds the required accuracy range.

````
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import glob
from PIL import Image

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
````

I trained the model on a dataset from [Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification), the dataset is linked. There are 6 categories of fruits: **Fresh apples**, **Fresh oranges**, **Fresh bananas**, **Rotten apples**, **Rotten oranges**, **Rotten bananas**.
This means that my model will require an output layer of 6 neurons to do the categorization successfully. And since I have multiple guesses I cannot use ` BCE `, I used ` Categorical_crossentropy `.

I started with a pretrained model from ImageNet, and I had to load it with the correct weights. Because these pictures are in color, there will be three channels for red, green and blue. Giving the model a new purpose.

````
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)
````
Then I freezed the base model, this is because I don't want to destroy the initial training (Making the whole idea of premodeling useless).
````
# Freeze base model
vgg_model.requires_grad_(False)
next(iter(vgg_model.parameters())).requires_grad
````
I want the more "general" learnings from VGG, so I selected the first 3 layers to the pretrained model.
````
vgg_model.classifier[0:3]
````
The 3 layers are Linear, reLU and Dropout. Those make sure it has the correct number of neurons to classify the different types of fruit.
````
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5, inplace=False)
)
````
At last, I added the following modifications
````
N_CLASSES = 6

my_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500, N_CLASSES)
)
my_model
````
In this modification, I transformed the images into features that the classifier could use by extracting features first, then reducing the spatial size using average pooling. Lastly, Flattening the images (turning them into a vector).
Then the orginal classifier from the premodel, and a 3 custom layers:
1- Linear layer reducing dimensionality.
2- ReLU for activation.
3- Another linear layer reducing dimentionality to 6 classes.

I compiled the model using loss and metrics functions, such as `CrossEntropyLoss()`. 
````
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(my_model.parameters())
my_model = torch.compile(my_model.to(device))
````
I preprocessed the input images by using transforms included in the VGG16 weights.
````
pre_trans = weights.transforms()
````
Then I augmented the data to improve the dataset, I used an example from the course.
````
IMG_WIDTH, IMG_HEIGHT = (224, 224)

random_trans = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.5)
])
````
I then loaded the train and validation datasets.

````
DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"] 
    
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []
        
        for l_idx, label in enumerate(DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)
            for path in data_paths:
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device))


    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)
````
Then selected 32 as a size for the batches, the reason I chose 32 was that I found out that most researches rely on 32 or 64. It is good to mention that the training dataset is shuffled, this is so the data is random and the model is able to train differently each timem, but the validation dataset is not, this is due to the unneedness of it to be random, you can edit it yourself but it is not needed and won't really matter, since the data is new anyways.
### Change the train and valid relative paths because otherwise it won't work, the relative path I used here is not the same as the one I uploaded into the repo.
````
n = 32

train_path = "data/fruits/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

valid_path = "data/fruits/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n, shuffle=False)
valid_N = len(valid_loader.dataset)
````
Then, I trained the model, I moded the train and validate functions to the utils.py file. You can check them there.
````
epochs = 10

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)
````

This should be more than enough to get more tthan 92% accuracy, but you can also try nad unfreeze the model for some fine tuning with a very low learning rate (.0000 or .00000).

Last, I validated the model by running the validate function in utils
````
utils.validate(my_model, valid_loader, valid_N, loss_function)
````

