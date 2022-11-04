# Autogating (gate 1 & 2) with U-Net
We use [U-Net](https://en.wikipedia.org/wiki/U-Net) to perform cytometry autogating for gate 1 and gate 2. The base code for U-Net training is adapted from [this repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation/unet). Please make sure you have Python 3.8+ and Pytorch 1.10+ to run the code.
## Setup and file structure
To use the code, please download this folder (not the entire repository, as we won't use the FCN) to the directory with the folder containing all the original csv files. Note that the folder containing the csv files is assumed to be named `omiq_exported_data_processed`, otherwise you need to change the code in `gen_image_label.py`, `general_utils.py`, `predict.py`, and `visualize.py` accordingly. Below is the correct placement:
```
.
└───omiq_exported_data_processed
│   │   01.T1_Normalized.csv
│   │   01.T2_Normalized.csv
│   │	...
│   
└───Unet
    |	gen_image_label.py
    |	general_utils.py
    |	my_dataset.py
    |	predict.py
    |	train.py
    |	visualize.py
    |
    └───src
    |   |    __init__.py
    |   |    unet.py
    │
    └───train_utils
    	|    __init__.py
    	|    dice_coefficient_loss.py
    	|    distributed_utils.py
    	|    train_and_eval.py
```
Most of the code uses relative path, except for one place. In `visualize.py`, make sure you change the `pred_root` and `label_root` (at the beginning of the main function) to fit your computer.  
</br>
**The file structure of this project:**  
- `src`: build the U-Net model
- `train_utils`: modules for training and validation
- `gen_image_label.py`: convert all the csv files to images and masks suitable for training
- `general_utils.py`: miscellaneous modules
- `my_dataset.py`: generate pytorch dataset
- `predict.py`: perform autogating
- `train.py`: train the model for autogating
- `visualize.py`: visualise the autogating results

## Workflow
We start by [converting all the csv files to images and masks](#Generate-images-and-labels) suitable for training. All the images and masks are generated within a big folder, thus the next step is to [divide them into train/validation/test datasets](#Create-train/validation/test-datasets). Note this procedure is done manually. After we have created the datasets, we can begin to [train the model](#Train-the-model). After we have trained the models for gate 1 and gate 2, we can then [perform autogating](#Autogate-the-raw-data) using the trained models. As an extra step, there are several plotting options to [visualise the autogating results](#Visualise-the-result). Finally, there are some [other utilities](#Other-utilities) to quantify the accuracy of the autogating results (when true labels are provided).

## Generate images and labels
The raw data has great precision, thus if we generate an image that truthfully represents all the cells, the resulting resolution will be too high for the network to process. Therefore, we need to compress the representation. This is how we do it:  
<table>
<tr>
<td colspan="2"> <b>Gate 1</b> </td>
<td colspan="2"> <b>Gate 2</b> </td>
</tr>
<tr>
<td>Ir191Di___191Ir_DNA1</td>
<td> x = x * 10000 / 622
<td>Ir193Di___193Ir_DNA2</td>
<td> x = x * 1000 / 43</td>
</tr>
<tr>
<td>Event_length</td>
<td>y = y - 10</td>
<td>Y89Di___89Y_CD45</td>
<td>y = y * 10000 / 330</td>
</tr>
</table>

After the above conversions, gate 1 images are 166x166 and gate 2 images are 256x256 (these images are saved as numpy arrays, i.e. .npy files). The images we use for training are density images. For each pixel, its value represents the number of cells landing at this pixel after the above conversion is applied. Below is an example of the actual density images we use for training.  

<img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/01.T1_gate1_density.png" alt="gate 1 density image" width="400"/><img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/01.T1_gate2_density.png" alt="gate 2 density image" width="400"/>

For the labels, axes are converted in the same way. The masks we use for training are binary images. For each pixel, its value is either 0 or 1, with 0 representing out-gate and 1 representing in-gate. Below is an example of the actual masks we use for training.

<img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/01.T1_gate1_label.png" alt="gate 1 mask" width="400"/><img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/01.T1_gate2_label.png" alt="gate 2 mask" width="400"/>

To generate the images and masks for all the raw data, run the following command (after `cd` into the Unet directory):
```
python gen_image_label.py
```
This command generates images and masks for both gate 1 and gate 2. In particular, it creates a new folder under the `Unet` directory, named `all_images`. The structure of this folder is shown below:
```
Unet
|	...
└───all_images
    └───gate_1
    |   └───image
    |	|    |    01.T1_Normalized.npy
    |	|    |    ...
    |    └───label
    |	|    |    01.T1_Normalized.npy
    |	|    |    ...
    └───gate_2
        └───image
	|    |    01.T1_Normalized.npy
	|    |    ...
	└───label
	|    |    01.T1_Normalized.npy
	|    |    ...
```
In addition, you can pass in the argument `-g 1` or `-g 2` (equivalently, `--gate 1` or `--gate 2`) to specify a gate. If `-g 1` is passed in, only images and masks for gate 1 will be generated, the same goes for gate 2. By default, images and masks for both gates will be generated.

## Create train/validation/test datasets
Once we have all the images and masks ready for training, the next step is to divide them into train/validation/test datasets. Since this is completely up to your choice, this step is done manually. Under the `Unet` directory, please create a folder of the following structure:
```
Unet
|	...
└───images
    └───gate_1
	|   └───train_image
	|   └───train_label
	|   └───val_image
	|   └───val_label
	|   └───test_image
	|   └───test_label
    └───gate_2
	    └───train_image
	    └───train_label
	    └───val_image
	    └───val_label
	    └───test_image
	    └───test_label
```
Select the images and the corresponding masks you want to use for validation or testing, and move them to the proper folders. Then move the rest of the images and masks to the folders labelled for training. Since this step is done manually, there is a chance that you might make a mistake. Don't worry, before training begins, there is a verification procedure that will tell you if you make any mistake.

## Train the model
We are now ready to train the models. Since the cells in gate 2 form a subset of those in gate 1, the density image and mask for gate 2 are not affected by gate 1. Therefore, we can train the autogating models for gate 1 and gate 2 separately. To train the autogating model for gate 1, simply run the following command under the `Unet` directory:
```
python train.py -g 1 --save-best
```
Similarly, you can run the following command to train the model for gate 2:
```
python train.py -g 2 --save-best
```
Before the training begins, there is a verification procedure to check it you did the previous step correctly. If there is any image without a corresponding mask (and vice versa), the verification procedure will stop and notify you of the mismatch. After the verification completes, the model will start to train. During the training process, basic information will be displayed and logged. This includes (for each epoch) loss value, learning rate, dice coefficient, global correct rate, class-wise correct rate, class-wise IOU value, and mean IOU value. Note that the dice coefficient, correct rate, and IOU value are obtained on the *validation* set. The training log is saved as a text file under the `Unet` directory, named `results{date}-{time}.txt`.  

After the training for both gates complete, a new folder will appear under the `Unet` directory, with the following structure:
```
Unet
|	...
└───saved_weights
    └───gate_1
    |	|	...
    └───gate_2
    |	|	...
```
In the folder `gate_1`, you will find the trained model for gate 1, and similar for gate 2.  

There are many additional arguments you can pass in:  
1. `-b` or `--batch-size`: set the size of each minibatch. default: 4
2. `--device`: set the training device. default: cuda
3. `--epochs`: set the number of training epochs. default: 50
4. `--print-freq`: set the frequency of info display. default: 10
5. `--lr`: set the initial learning rate. default: 0.01
6. `--momentum`: set the descent momentum. default: 0.9
7. `--wd` or `--weight-decay`: set the weight decay rate. default: e-4
8. `--resume`: resume from a previous training
9. `--start-epoch`: set the start epoch. default: 0
10. `--save-best`: save only the best model
11. `--amp`: use mixed precision training

In my experience, the default hyper-parameters train pretty well.

### Save the model
The model is saved automatically, and you can find it (.pth file) in the `saved_weights` folder. In the training command, you should specify whether you want to save only the best model or all the models (one for each epoch). If you want to save only the best model, pass in the argument `--save-best`. If you want to save a model for every epoch, pass in the argument `--no-save-best`.

## Autogate the raw data
**NOTE: you need to train both gates before autogating!**  
After training is completed, we can then perform autogating on raw data. Run the following command under the `Unet` directory:
```
python predict.py -f filename --weights-gate1 weights1 --weights-gate2 weights2
```
`filename` should be a string, e.g. "01.T1_Normalized". `weights1` and `weights1` should be the models you wish to use. By default, they are both set to "best_model.pth", thus if you chose `--save-best` during the training, you don't have to specify them in the above command. However, if your model has a different name, please pass in the correct argument `--weights-gate* "?.pth"`. The model for gate 1 should be put in the directory `Unet/saved_weights/gate_1`, and that for gate 2 in `Unet/saved_weights/gate_2`.  

### The input file
The input file should be a .csv file with at least four columns:
`Ir191Di___191Ir_DNA1`, `Event_length` for gate 1 prediction, and `Ir193Di___193Ir_DNA2`, `Y89Di___89Y_CD45` for gate 2 prediction. Any extra columns will be ignored.

### The output file
The first time you run the `predict.py` programme, a new folder named `prediction_results` will be created under the `Unet` directory. All subsequent autogating results will be saved in this folder. After the network autogates the input file, a .csv file will be generated and saved under the `Unet/prediction_results` directory. It will be named "prediction__{filename}.csv". For example, if you autogate the input file "01.T1_Normalized.csv", then the autogating results will be saved in the output file "prediction__01.T1_Normalized.csv". The output file contains six columns:
| Ir191Di___191Ir_DNA1 | Event_length | Ir193Di___193Ir_DNA2 | Y89Di___89Y_CD45 | gate1_ir | gate2_cd45 |
| --- | --- | --- | --- | --- | --- |
|6.9935 | 36 | 7.5951 | 5.9978 | 1 | 1 |
|5.5613 | 26 | 6.0384 | 2.326 | 0 | 0 |
| 7.9067 | 32 | 8.4679 | 5.4671 | 1 | 0 |
| ... | ... | ... | ... | ... | ... |

The first four columns are the same as the input file. The fifth column `gate1_ir` is the autogating result for gate 1, and `gate2_cd45` is the autogating result for gate 2.  

**NOTE: although we trained the model for gate 1 and gate 2 separately, when performing autogating, we multiply the predicted value for gate 2 by the predicted value for gate 1, so that the cells predicted to be in gate 2 are guaranteed to be a subset of those predicted to be in gate 1.**

## Visualise the result
We have provided several options to visualise the autogating results.  

**Note: `pred_root` and `label_root` in the main function of `visualize.py` should be changed to fit your computer.**

### Compare with the true gate
If you have true gate (e.g. autogate a test file), you can compare the autogating result with the true gate by running the following command under the `Unet` directory:
```
python visualize.py -f filename --compare
```
Make sure the autogating result you are visualising is saved as `Unet/prediction_results/prediction__{filename}.csv`. You can pass in an optional argument `-g 1` or `-g 2` to visualise only gate 1 or gate 2.  If you don't pass in this argument, both gates will be visualised. An example plot is shown below.

<img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/02.T1.png" alt="visualise autogating result for both gates" width="1000"/>

There is another optional argument you can pass in, `--filter-gate1`. When this argument is passed in, the visualisation for gate 2 will not show all the cells, but only those in gate 1.

### Original resolution
If you don't have true gate, or you don't want to compare with the true gate, you can visualise the autogating result for gate 1 and gate 2 side-by-side, at original resolution. Run the following command under the `Unet` directory:
```
python visualize.py -f filename --no-compare
```
An example plot is shown below.

<img src="https://github.com/MapleBKL/cytometry-autogating/blob/main/Unet/readme_pictures/02.T1-no-compare.png" alt="visualise autogating result for both gates" width="800"/>

Red cells are out of the gate, blue cells are in the gate. The black polygon shows the convex hull of the predicted gate.

## Other utilities
Finally, there are some additional utilities for quantifying and visualising. To use these utilities, you need to first enter the interactive environment of `general_utils.py`. Run the command
```
python -i general_utils.py
```
Three functions are currently available: `compute_iou`, `compute_dice`, and `plot_diff`. You need to first set the root and import `matplotlib.pyplot`. Run the following commands
```
>>> root = "D:\\Cytometry\\omiq_exported_data_processed"  # change this to fit your computer
>>> import matplotlib.pyplot as plt
```
`compute_iou` computes the IOU (intersection over union) index of the autogating result. Let $P_i$={predicted in gate $i$} and $A_i$={actual in gate $i$}, then the IOU index is computed as
```math
\text{IOU}_i = \frac{|P_i\cap A_i|}{|P_i\cup A_i|}
```
Run the command
```
>>> compute_iou(filename)
```
will print the IOU factor for the autogating result of the specified file.  

`compute_dice` computes the dice coefficient of the autogating result. The dice coefficient is computed as
```math
\text{dice}_i = \frac{2\times |P_i\cap A_i|}{|P_i|+|A_i|}
```
Run the command
```
>>> compute_dice(filename)
```
will print the dice coefficient for the autogating result of the specified file.  

Finally, `plot_diff` plots the difference image between the predicted gate and the actual gate. Run the command
```
>>> plot_diff(filename, gate)
```
to plot the difference image for the specified file and specified gate.
