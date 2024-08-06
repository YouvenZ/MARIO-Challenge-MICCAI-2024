import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from utils.data_processing import preprocess_input
from dataloader import SimpleDataset
from utils.scoring import specificity
from tqdm import tqdm
from models.example_model_task1 import SimpleModel1
from models.example_model_task1v2 import SimpleModel1v2

from torch.utils.data import DataLoader 
import torchvision.transforms as T


import operator
import functools

image_size = 256

data_transforms = {
    'train': T.Compose([
        T.RandomRotation(15),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(image_size, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        T.ToTensor(),
    ]),
    'val': T.Compose([
        T.Resize([image_size,image_size]),
        T.ToTensor(),
    ]),
    'test': T.Compose([
        T.Resize([image_size,image_size]),
        T.ToTensor(),
    ])
}




class InferenceTask1:
    def __init__(self, model_paths, model_names,model_weights=None,*args, **kwargs):
        """
        Initializes the inference class with model paths and weights.

        Args:
            model_paths (list): List of paths to the model files.
            model_weights (list, optional): List of weights for each model. Defaults to equal weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [self.load_model(model_name,model_path) for model_name,model_path in zip(model_names,model_paths)]
        print(f"Using device: {self.device}")
        self.i = 0
        if model_weights is None:
            self.model_weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            self.model_weights = model_weights

    def load_model(self, model_name, model_path,*args, **kwargs):
        """
        Loads a model from a given path and it's class name.

        Args:
            model_name (str): name of the model class.
            model_path (str): Path to the model file.
            

        Returns:
            torch.nn.Module: Loaded model.
        """
        
        model = eval(model_name)(*args, **kwargs)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def simple_inference(self, data_loader):
        """
        Performs inference on the data using the loaded model.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        ## The proposed example only use the pair of OCT slice, but you are free to update if your pipeline involve
        ## localizer and the clinical, udapte accordingly 
        
        y_true = []
        y_pred = []
        cases = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                
                oct_slice_xt0 = batch["oct_slice_xt0"].to(self.device)
                oct_slice_xt1 = batch["oct_slice_xt1"].to(self.device)
                localizers_xt0 = batch["localizers_xt0"].to(self.device)
                localizers_xt1 = batch["localizers_xt1"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                

                inputs = torch.concat([oct_slice_xt0, oct_slice_xt1], dim=1)
                output = self.models(inputs)
                prediction = output.argmax(dim=1).item()
                y_pred.append(prediction)
                y_true.append(label)
                cases.append(case_id)
        return y_true, y_pred, cases
    
    
    
    def scoring(self, y_true, y_pred):
        """
        DO NOT EDIT THIS CODE
        Calculates various scoring metrics.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary containing various scores.
        """
        return {
            "F1_score": f1_score(y_true, y_pred, average="micro"),
            "Rk-correlation": matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
        }
    

    def test_time_augmentation(self, input_tensor, n_augmentations=5):
        """
        Applies test time augmentation to the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            n_augmentations (int): Number of augmentations to apply.

        Returns:
            torch.Tensor: Averaged output from the model after applying augmentations.
        """
        augmentations = T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        
        outputs = []
        for _ in range(n_augmentations):
            augmented_input = augmentations(input_tensor).to(self.device)
            with torch.no_grad():
                outputs.append(self.models[0](augmented_input))
        return torch.mean(torch.stack(outputs,dim=0), dim=0)
                    
                    
                    
                

    def simple_ensemble_inference(self, data_loader):
        """
        Performs inference using model ensembling and test time augmentation.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        y_true = []
        y_pred = []
        cases = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                
                oct_slice_xt0 = batch["oct_slice_xt0"].to(self.device)
                oct_slice_xt1 = batch["oct_slice_xt1"].to(self.device)
                localizers_xt0 = batch["localizers_xt0"].to(self.device)
                localizers_xt1 = batch["localizers_xt1"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                                


                inputs = torch.concat([oct_slice_xt0, oct_slice_xt1], dim=1)
                outputs = []
                for model, weight in zip(self.models, self.model_weights):
                    output = model(inputs)
                    outputs.append(output * weight)
                averaged_output = torch.mean(torch.stack(outputs), dim=0)
                prediction = list(averaged_output.argmax(dim=1).cpu().detach())
                
                y_pred.append(prediction)
                y_true.append(label)
                cases.append(case_id)

        return y_true, y_pred, cases

    def simple_test_time_inference(self, data_loader, n_augmentations=5):
        """
        Performs inference with test time augmentation and model ensembling.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        y_true = []
        y_pred = []
        cases = []

        with torch.no_grad():
            
            for batch in tqdm(data_loader):
                
                oct_slice_xt0 = batch["oct_slice_xt0"].to(self.device)
                oct_slice_xt1 = batch["oct_slice_xt1"].to(self.device)
                localizers_xt0 = batch["localizers_xt0"].to(self.device)
                localizers_xt1 = batch["localizers_xt1"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                tta_outputs = []
                for _ in range(n_augmentations):
  
                    inputs = torch.concat([oct_slice_xt0, oct_slice_xt1], dim=1)
                    augmented_inputs = self.test_time_augmentation(inputs, n_augmentations)

                    prediction = list(augmented_inputs.argmax(dim=1).cpu().detach())
                    

                    
                    y_pred.append(prediction)
                    y_true.append(label)
                    cases.append(case_id)

        return y_true, y_pred, cases

    def run(self, data_loader, use_ensemble = True, use_tta=False, n_augmentations=5):
        """
        Runs the inference and saves results.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            use_tta (bool): Whether to use test time augmentation.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            dict: Dictionary containing various scores.
        """
        
        
        ## You can test as much inference pipeline you which
        # in your local machine. You will have to select
        # two shot to for the final submission. 
        # The inference should always return a list of batch containing label,prediction,cases 
        # The method run should always return the scores
        
        if use_tta:
            y_true, y_pred, cases = self.simple_test_time_inference(data_loader, n_augmentations)
        elif use_ensemble:
            y_true, y_pred, cases = self.simple_ensemble_inference(data_loader)
        
        #### elif:
            # Any custom inference that you want to apply
        
        else:
            y_true, y_pred, cases = self.simple_inference(data_loader)
            
            


        # DO NOT EDIT THIS PART


        y_true = functools.reduce(operator.iconcat, y_true, [])
        y_pred = functools.reduce(operator.iconcat, y_pred, [])
        cases = functools.reduce(operator.iconcat, cases, [])

        output_file = f"output/results_task1_team_{os.environ['Team_name']}_method_{self.i}.csv"
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'cases': cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i +=1
        return self.scoring(y_true, y_pred)

# Main execution
print(f"Starting the inference for the team: {os.environ['Team_name']}")

# Load data
dataset = SimpleDataset('csv/df_task1_val.csv', 'data/', transform=data_transforms['test'])
data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

model_paths = ['models/model_task1.pth', 'models/model_task1v2.pth']  # Example for multiple models
model_names = ["SimpleModel1", "SimpleModel1v2"]
model_weights_contribution = [0.6, 0.4]  # Example weights for the models
inference_task1 = InferenceTask1(model_paths, model_names, model_weights_contribution)



scores_1 = inference_task1.run(data_loader, use_tta=True, n_augmentations=5)
print(f"Obtained scores for inference method 1: F1_score: {scores_1['F1_score']}, Rk-correlation: {scores_1['Rk-correlation']}, Specificity: {scores_1['Specificity']}")

scores_2 = inference_task1.run(data_loader,use_ensemble = True, use_tta=False)
print(f"Obtained scores for inference method 2: F1_score: {scores_2['F1_score']}, Rk-correlation: {scores_2['Rk-correlation']}, Specificity: {scores_2['Specificity']}")
