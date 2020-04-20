from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import pandas as pd
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split + '.txt' # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt'
            
        self.split_path = os.path.join(self.root.split('/')[0], self.split) #train.txt file path
            
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        label_counter  = 0
        labels_dict ={}
        labels = []
        img_paths = []
           
        paths = np.loadtxt(self.split_path, dtype=str) #array with paths in the form 'class_name/imagexxxx.jpg'
        for path in paths:
            fields = path.split('/') #fields[0] = class_name
            if fields[0]!='BACKGROUND_Google': #drop BACKGROUND_Google folder
                if fields[0] in labels_dict: #if label already met
                    labels.append(labels_dict[fields[0]]) #assign corresponding label
                    img_paths.append(path) #assign corresponding image path
                else:
                    labels_dict[fields[0]] = label_counter; #add new label to the dictionary
                    labels.append(label_counter); #assign corresponding label
                    img_paths.append(path) #assign corresponding image path
                    label_counter += 1 #increment label counter
                
        self.dataset = pd.DataFrame({'path': img_paths, 'label': label})
        self.length = len(self.dataset)

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image = pil_loader(os.path.join(self.split_path, self.dataset.iloc[ind, 0]))
        label = self.dataset.iloc[ind, 1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        
        '''
        length = ... # Provide a way to get the length (number of elements) of the dataset
        return length
        
        '''
        return self.length
