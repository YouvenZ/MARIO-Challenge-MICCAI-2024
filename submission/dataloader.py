import pandas as pd
import numpy as np
from PIL import Image
import os




import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms




class SimpleDataset(Dataset):
    def __init__(self, csv_file, root_dir, task=1, transform=None):
        self.data = pd.read_csv(csv_file).head(150) # only 150 rows for debugging
        self.data["sex"] = self.data["sex"].apply(lambda x: 0 if x=='M' else 1) 
        
        # Good place to add normalization for time
        
        self.root_dir = root_dir
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        case_id = self.data.iloc[idx]["case"]
        label = self.data.iloc[idx]["label"]
        
        if self.task == 1:
            
            oct_slice_xt0_path = self.data.iloc[idx]["image_at_ti"]
            oct_slice_xt1_path = self.data.iloc[idx]["image_at_ti+1"]
            localizers_xt0_path =self.data.iloc[idx]["LOCALIZER_at_ti"]
            localizers_xt1_path = self.data.iloc[idx]["LOCALIZER_at_ti+1"]
            clinical_data = self.data.iloc[idx][['sex','age_at_ti+1','age_at_ti','num_current_visit_at_i+1','num_current_visit_at_i','delta_t']].values
            label = self.data.iloc[idx]["label"]

            
            oct_slice_xt0 = Image.open(os.path.join(self.root_dir,oct_slice_xt0_path)).convert('RGB')
            oct_slice_xt1 = Image.open(os.path.join(self.root_dir,oct_slice_xt1_path)).convert('RGB')
            localizers_xt0 = Image.open(os.path.join(self.root_dir,localizers_xt0_path)).convert('RGB')
            localizers_xt1 = Image.open(os.path.join(self.root_dir,localizers_xt1_path)).convert('RGB')
            
            if self.transform:
                                
                oct_slice_xt0 = self.transform(oct_slice_xt0)
                oct_slice_xt1 = self.transform(oct_slice_xt1)
                localizers_xt0 = self.transform(localizers_xt0)
                localizers_xt1 = self.transform(localizers_xt1)

            sample['oct_slice_xt0'] = oct_slice_xt0
            sample['oct_slice_xt1'] = oct_slice_xt1
            sample['localizers_xt0'] = localizers_xt0
            sample['localizers_xt1'] = localizers_xt1

        else:
            oct_slice_xt_path = self.data.iloc[idx]["image"]
            localizers_xt_path =self.data.iloc[idx]["LOCALIZER"]
            clinical_data = self.data.iloc[idx][['sex','age','num_current_visit']].values


            
            oct_slice_xt = Image.open(os.path.join(self.root_dir,oct_slice_xt_path)).convert('RGB')
            localizers_xt = Image.open(os.path.join(self.root_dir,localizers_xt_path)).convert('RGB')

            if self.transform:
                oct_slice_xt = self.transform(oct_slice_xt)
                localizers_xt = self.transform(localizers_xt)
                
            sample['oct_slice_xt'] = oct_slice_xt
            sample['localizers_xt'] = localizers_xt

        sample['case_id'] = int(case_id)
        sample['clinical_data'] = [int(e) for e in clinical_data]
        sample['label'] = int(label)
        return sample


