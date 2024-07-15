import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import random


class Augment:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
            # Add more transformations here
        ])

    def __call__(self, image, mask):
        # Apply the same random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # Apply the same random vertical flip
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Apply rotation
        if random.random() > 0.5:
            image = F.rotate(image, 10)
            mask = F.rotate(mask, 10)
        
        # Convert to tensor
        image = self.image_transform(image)
        mask = self.image_transform(mask)

        return image, mask

class CustomDataset(Dataset):
    def __init__(self, root_dir,mask_dir, train_flag, augment=False):
        """
        Set the root directory , get list of images
        transform to tensor and normalize the data
        """
        print("Initialize Custom Data")
        self.root_dir = root_dir
        self.train_flag = train_flag
        #if train_flag:
        self.mask_dir = mask_dir
        self.mask_list = [file for file in os.listdir(mask_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

        self.image_list = [file for file in os.listdir(root_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,)),  # Adjust normalization as needed
            # mean and standard deviation tuple sent as parameter for normalization
            transforms.Resize((256, 256), antialias=True)
        ])

        # check if augmentation is true
        if augment:
            print('Augmentation Initialized')
            self.augment = Augment()
        else:
            self.augment = augment

    def __len__(self):
        if self.augment:
            return len(self.image_list)*2
        else:
            return len(self.image_list)
        

    def __getitem__(self, idx):
        # check the original idx and check if this is an augmented sample
        if self.augment:
            original_idx = idx // 2
            augment_flag = idx % 2 == 1
        else:
            original_idx = idx
            augment_flag = False

        img_name = os.path.join(self.root_dir, self.image_list[original_idx])
        image = Image.open(img_name).convert('L') # convert to gray scale
        
        #if self.train_flag:
        mask_name = os.path.join(self.mask_dir, self.image_list[original_idx])  # Assuming masks have the same filenames
        mask = Image.open(mask_name).convert('L')
        
        # Apply augmentation if needed
        if augment_flag and self.augment:
            image, mask = self.augment(image, mask)
            augmented = True
        else:
            image = self.transform(image)
            mask = self.transform(mask)
            augmented = False
       
        #mask = self.transform(mask) 

        #image = self.transform(image)
        
        #print("getItem:: image length::",image.shape)
        #print("getItem:: mask length::",mask.shape)

        # if self.train_flag:
        #     return image, mask
        # else:
        return image, mask