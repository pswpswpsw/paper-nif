# Section 3.1: Data-fit mesh-agnostic surrogate modeling

## Data generation

Both training and testing data has been generated already in `DATA_FINAL`
```bash
cd DATA_FINAL
ls
```
Details of how the data is generated can be found in `DATA_FINAL/README.md`

## Compare different model configuration given 20 simulation data

### Data

here we use `DATA_FINAL/ks_train_n_20.npz` as our training data, and `DATA_FINAL/ks_test_n_59.npz` as testing data

### Training

- go to the folder `ks_0`
```bash
cd ks_0
```

- baseline configuration is using swish and 20 K-S simulations. it is in `train_20_swish`
```bash
cd train_20_swish
```

- then go to each folder `relu`, `swish`, `tanh`, you will find `train_datafit_surrogate.py`

- that code is a single self contained python script

- in each subfolder where `train_datafit_surrogate.py` exist, run the following command. note the arguments are self-evident.

1. MLP-Swish
```bash
python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
```

2. NIF-Swish
```bash
python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 56 --N_T 30 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
```

3. MLP-tanh
```bash
python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT tanh --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
```

4. NIF-tanh
```bash
python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 56 --N_T 30 --ACT tanh --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
```

5. MLP-ReLU
```bash
python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT relu --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
```

Note: for each case, remember to run 4 converged runs, since sometimes it is possible to have unconverged mini-batched training, which makes the comparison invalid. one could reduce this problem by early stopping but here for the sake of simplicity, just run to have 4 converged runs.

### Test and Visualization

when given 20 simulation data as training data, we can visualize the difference among these aboev model configurations.
```bash
jupyter notebook appendix_C1_datafit_for_1d_ks.ipynb
cd paper-pngs
```

## Validate model performance by changing model parameters

### Data

still, we use the same training data as before, the one with 20 simulations. thus, the data is in `DATA_FINAL/ks_train_n_20.npz`. model performance is test on `DATA_FINAL/ks_test_n_59.npz`.

### Training

- go to `ks_0/change_mp_swish_train_20`
- each subfolder represents the amount of model parameters. 
- `1_less`

  - NIF, 6935 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 30 --N_T 30 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

  - MLP, 7135 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 58 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

- `1.5_lm`

  - NIF, 10254 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 38 --N_T 29 --ACT swish --L_R
	```
  
  - MLP, 10291 parameters 		

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 70 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

- `2.5_mh`

  - NIF, 24966 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 60 --N_T 47 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

  - MLP, 24971 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 110 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

- `3_high`

  - NIF, 34415 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../ks_train_n_20.npz --NETWORK_TYPE NIF --N_S 70 --N_T 60 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```

  - MLP, 34711 parameters

	```bash
	python train_datafit_surrogate.py --TRAIN_DATA ../../../ks_train_n_20.npz --NETWORK_TYPE MLP --N_S 130 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
	```


### Test and Visualization

Finally, we validate the model performance on testing data and visualize the results. To do so, simply go to the folder where we have `plot_change_model_parameters.ipynb`

```bash
jupyter notebook plot_change_model_parameters.ipynb
cd paper-pngs
```

## Validate model performance by changing the amount of training data

### Data

Now we will varying the size of training data. Data is in `DATA_FINAL/ks_train_n_15.npz`, `DATA_FINAL/ks_train_n_24.npz` and `DATA_FINAL/ks_train_n_29.npz`.

### Training

- go to folder `ks_0`, you can find that we have three additional folders associated with this study. they are `train_15_swish`, `train_24_swish`, `train_29_swish`.
- go deep to each of the above folders until you find `train_datafit_surrogate.py` and then run the following commands
	- `train_15_swish`
	  - MLP
		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_15.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

	  - NIF

		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_15.npz --NETWORK_TYPE NIF --N_S 56 --N_T 30 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

	- `train_24_swish`

	  - MLP

		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_24.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

	  - NIF

		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_24.npz --NETWORK_TYPE NIF --N_S 56 --N_T 30 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

	- `train_29_swish`

	  - MLP

		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_29.npz --NETWORK_TYPE MLP --N_S 100 --N_T 0 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

	  - NIF
	  
		```bash
		python train_datafit_surrogate.py --TRAIN_DATA ../../../../DATA_FINAL/ks_train_n_29.npz --NETWORK_TYPE NIF --N_S 56 --N_T 30 --ACT swish --L_R 0.001 --BATCH_SIZE 1024 --EPOCH 40001
		```

### Test and Visualization

now since you have run NIF and MLP with the same number of trainable parameters for 4 different amount of simulation data, we can compare the RMSE between MLP and NIF for those cases as well. 
```bash
jupyter notebook plot_change_training_data.ipynb
cd paper-pngs
```






