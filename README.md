# Vision Transformer for AutoML Tasks

This repository presents a Vision transfomer and Gradient Boosting model for automl tasks.

## Pre-requisites
Before running this scripts, make sure to install the required modules. Navigate to the requirements.txt file and run the code below in your suitable terminal environment.  

```
pip install -r requirements.txt
```
or 
```
python -m pip install -r requirements.txt
```
Also make sure to use python version 3.9 and create a virtual env to run the above scripts.

## Criterion
| Model Type                  | Task                                |
|-----------------------------|-------------------------------------|
| Transformer with time-based tokenization | Sequence and signal data analysis  |
| Vision Transformer          | Image classification               |
| U-Net                       | Image segmentation                 |
| LightGBM                    | Tabular data analysis              |

## Training and Testing
The model is designed to take in metadata to configure the performance metrics and output task. Identify the directory path of your dataset and configure the path variables in th  ```main.py``` file.

To test the model, test dataset is hosted on [kaggle](https://kaggle.com/datasets/f006b553d95f1ddf163f59e2220dbf3b49db42dbfa76428883e42e0036c044e5). You can download the entirety or specific to dataset description.
