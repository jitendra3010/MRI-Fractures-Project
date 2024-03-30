import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir,mask_dir, train_flag):
        """
        Set the root directory , get list of images
        transform to tensor and normalize the data
        """
        print("Initialize Custom Data")
        self.root_dir = root_dir
        self.train_flag = train_flag
        if train_flag:
            self.mask_dir = mask_dir
            self.mask_list = [file for file in os.listdir(mask_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

        self.image_list = [file for file in os.listdir(root_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Adjust normalization as needed
            # mean and standard deviation tuple sent as parameter for normalization
            transforms.Resize((256, 256), antialias=True)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('L') # convert to gray scale
        
        if self.train_flag:
            mask_name = os.path.join(self.mask_dir, self.image_list[idx])  # Assuming masks have the same filenames
            mask = Image.open(mask_name).convert('L')
            mask = self.transform(mask) 

        image = self.transform(image)
        
        #print("getItem:: image length::",image.shape)
        #print("getItem:: mask length::",mask.shape)

        if self.train_flag:
            return image, mask
        else:
            return image