# features/xray_predictor/model.py
import torch
import torch.nn as nn
from torchvision import models

# A list of the 14 disease labels that our model predicts
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def create_model():
    """
    Creates and returns the DenseNet-121 model with the correct architecture.
    """
    # Load the DenseNet-121 architecture, but without any pre-trained weights
    model = models.densenet121(weights=None)
    
    # Get the number of input features for the classifier
    num_features = model.classifier.in_features
    
    # Replace the classifier with our custom one for 14 diseases
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(DISEASE_LABELS))
    )
    
    return model
