from automl.model import Model
from automl.extras import AutoDataset, Metadata
from automl.metrics import *

import torch
from torch.utils.data import DataLoader
from ingestion.dev_datasets import DecathlonDataset, extract_metadata

from scoring.score import get_solution, decathlon_scorer

dataset = "crypto"  # Replace with the desired dataset name
data_path = r"D:/Train Data/dev_public"

# Load the training and testing datasets using DecathlonDataset
train_dataset = DecathlonDataset(dataset, data_path, 'train')
test_dataset = DecathlonDataset(dataset, data_path, 'test')

# Extract metadata
md_train = extract_metadata(train_dataset)
md_test = extract_metadata(test_dataset)

# Print metadata for verification
print("Dataset path: ", md_train.get_dataset_name())
print("Input shape: ", md_train.get_tensor_shape())
print("Output shape:", md_train.get_output_shape())
print("Dataset size: ", md_train.size())

#Quick Solution
solution = get_solution(data_path, dataset)
print(solution.shape)
print(solution)

def get_dataloader(dataset, batch_size, split):
    """Get the PyTorch dataloader.
    Args:
        dataset:
        batch_size : batch_size for training set

    Return:
        dataloader: PyTorch Dataloader
    """
    if split == "train":
        dataloader = DataLoader(
            dataset,
            dataset.required_batch_size or batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )
    elif split == "test":
        dataloader = DataLoader(
            dataset,
            dataset.required_batch_size or batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
    return dataloader


batch_size = 1
train_loader = get_dataloader(train_dataset, batch_size, 'train')
test_loader = get_dataloader(test_dataset, batch_size, 'test')

# Initialize the model with training metadata
model = Model(md_train)

# Train the model
model.train(train_dataset, remaining_time_budget=0.05 * 3600)

# Test the model
yp = model.test(test_dataset)

# Prediction
prediction = model.test(test_dataset, remaining_time_budget=200)
print(prediction.shape)
print(prediction[0])

# Calculate and print the accuracy
score = decathlon_scorer(solution, prediction, dataset)
print ("Score: ", score)
