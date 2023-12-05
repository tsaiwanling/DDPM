# DDPM
Denoising Diffusion Probabilistic Models

## Tools

`utilits_lib`
* `diffusion_model.py`<br>
	To combine the forward_process(add noise) and reverse_process(remove noise).<br>
	Functions:
	1. `train(dataset, lr:float=0.01, epochs:int=20)`<br>
		To train the DDPM via forward_process.<br>
		* Optimizer: Adam.
		* Loss function: MSE loss for noises.
	2. `generator()`
		To randomly generate a graph by trained DDPM.

* `forward_process.py`<br>
	Adding noise.<br>
	* beta schedule: linear from 1e-5 to 0.02
* `reverse_process.py`<br>
	Here using <b>Unet</b> model to predict the noise.

## Usage
### Data Preparing
You can prepare the dataset by referring to `DataProcess.ipynb`, then save the dataset as .pkl file.<br>

### Training
To train a diffusion model, using `train.py` 

`python train.py --DataPath "./dataProcessed/YourData.pkl" --ImgSize 32 --TimeSteps 300 --LearningRates 1e-2 --Epochs 3000 --BatchSize 8 --Model_name "YourDDPM"`

|Parameters|Defalut|Description|
|---|---|---|
|DataPath|no Defalut|The training dataset, only for .pkl file.|
|ImgSize|320|This DDPM only valid for fixed square image.|
|TimeSteps|300|How many steps the forward process to add noise.|
|LearningRates|1e-2|Learning rate for Adam optimizer|
|Epochs|3000|The one entire passing of training data through the algorithm.|
|BatchSize|8|The number of units manufactured in a production run.|
|Model_name|no Defalut|Save the model as .pt file with Model_name|
