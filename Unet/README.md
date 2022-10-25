# Pre-gating (gate 1 & 2) with Unet

This folder contains several python files for training, predicting, and visualising.

The src folder contains the Unet model source code, and the train_utils folder contains auxiliary files for training. They should not be changed.

## Train
I have pretrained a model, and the weights can be found at the following link:  
https://drive.google.com/file/d/1MT8Sw9q1XY_OgcGI7Y1f1cXq5WlTq_dP/view?usp=sharing  

To use this pretrained model, download the weights (a .pth file) and create a new folder named "saved_weights" in the Unet folder. Put the downloaded weights file in the saved_weights folder.  

To train a new model, run the following command in the Unet folder:  
```python train.py```  

There are several arguments you can pass in, for example:  

**epochs**: change the training epochs. Default is 200.  
**batch-size**: change the size of each batch. Default is 4.  
**save-best**: when set to true, the programme only saves the model with the best validation result; when set to false, the programme saves the model of each epoch. Default is True.  

There are other parameters you can pass in, check the code for details.  

To create a proper dataset for training, run the following command in the Unet folder:  
```python create_train_set.py --val-size 5```  

You can change the validation set size by passing in different values for the parameter `val-size`. The script randomly picks `val-size` files for validation.  
This script creates a folder named "images" in the Unet folder, and in the images folder, it creates another four folders: "train_image", "train_label", "val_image", "val_label".  


## Predict
To make a prediction, run the following command in the Unet folder:  
```python predict.py -i image_filename -w weight_filename```  
or  
```python predict.py -f csv_filename -w weight_filename```  

You should provide either an image (a .npy file) or a .csv file, not both. `weight_filename` is the name of the trained model you wish to use. Default is "best_model.pth".  

When provided with an image, the programme will output an image (a .npy file) recording the predicted gate.  
When provided with a csv file, the programme will output an image recording the predicted gate, as well as a csv file with a predicted label for each cell.  
The output csv file has three columns:  
| Ir191Di___191Ir_DNA1 | Event_length | pred_gate_1 |
| --- | --- | --- |
| 4.5795 | 25 | 1 |
|... | ... | ... |
