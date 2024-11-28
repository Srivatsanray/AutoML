from automl.model import Model
from automl.extras import AutoDataset, Metadata
from automl.metrics import *

import torch
from ingestion.dev_datasets import DecathlonDataset, extract_metadata
dataset = 'nottingham'  # Replace with the desired dataset name
data_path = "/d/Train Data/dev_public/"

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

# Initialize the model with training metadata
model = Model(md_train)

# Train the model
model.train(train_dataset, remaining_time_budget=0.05 * 3600)

# Test the model
yp = model.test(test_dataset)

# Calculate and print the accuracy
print('Accuracy: {:.1%}'.format(
    1 - zero_one_error(
        yp, torch.stack([e[1] for e in test_dataset])
    )
))
