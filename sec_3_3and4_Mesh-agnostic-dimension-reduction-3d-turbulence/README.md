# Section 3.3: Mesh-agnostic compressed representation of 3D homogeneous isotropic turbulence

## Data preparation

- we have prepared the normalized training dataset: `3dhit-txyz-128-20-uvw.npz` (2.3 GB)
- if you still want to create the `npz` by yourself, you can
	- checkout the dataset by specifying the dataset up download JHU Turbulence dataset from the [link](http://turbulence.pha.jhu.edu/)
	- once the `isotropic1024coarse.h5` and corresponding `xmf` file is in the `./DATA`, run the following command 
	```bash
python prepare_3dhit_training_data.py
	```

- to compare against other NeRF related frameworks, we also prepared a smaller dataset, which is only 2D: `2dhit-txy-128-20-uvw.npz` (10.5 mb). Make sure you have put this file into the `./DATA` folder.

## Training
- go to `run` folder.
	```bash
	cd run
	```
- run the code (the default setup runs on a single GPU with a `batch_size=1600`), you should increase batch_size to the maximal capacity of your GPU card. if memory allocation error comes out, you need to reduce the `batch_size`.
	```bash
	python run_nif_txyz_siren_hit_mlp-pn-uvw.py --TRAIN_DATA ../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 200 --LAYER_HIDDEN_SPACE 4 --OMEGA_X 30.0 --NUM_HIDDEN_TIM 50 --LAYER_HIDDEN_TIME 1 --RANK_PARA 3 --NUM_SNAP 20 --BATCH_SIZE 1600 
	```
- training loss is plotted inside the model as `minibatchloss.png`. if the error is below 1.0 and steadily decreasing, then the training is good. otherwise, if the error is around 0.99, it means the training is pretty bad. 
- you can change the number of steps for printing out figures, saving checkpoints in the python script `run_nif_txyz_siren_hit_mlp-pn-uvw.py`.

## Inference on trained model to output `npz`
- now the model should be well trained ( by just checking the `log` file or `minibatchloss.png` ). you can take a look at the size of trained model checkpoint, without any pruning at all, the model is just `18.6 MB`  (compared to original H5 file with `504 MB`) and this is not yet optimized. then we can make model inference to see if this model performance is good enough given such **compressed representation**.

- you can go to the file `inference_get_npz.py` in `run` directory. it contains the specification for which model to inference and which checkpoint to use. 
```bash
python inference_get_npz.py
```
the output of this file is `output_arr_u.npz`, which is inside the model directory. 

## Write `npz` data back to `h5`

- in order to visualize the predicted 3D velocity field, we need to map it back to `h5` with a corresponding `xmf`. the easy way is just to modify an existing pair of `h5` and `xmf`, which comes from the original data of JHU dataset. however, `pred.h5`, `pred.xmf` and `true.h5`, `true.xmf` are prepared. so you just need to modify their values through `h5py`.

- `get_h5_from_npz.py` contains set up of which `npz` to choose, you can take a look so you will know how to modify it.
- run the following commands to generate `pred.h5` and `true.h5`
```bash
python get_h5_from_npz.py PRED
python get_h5_from_npz.py TRUE
```

## Visualize
- now you can safely leverage `ParaView` to visualize the model prediction by just reading these HDF5 files with `xmf`. you can compute Q criterion, draw iso-contours. a guide on how to use `ParaView` can be found [here](https://www.xsede.org/documents/234989/378230/VIS_PView_0212.pdf).
- then, we use `ParaView` to obtain the histogram of velocity magnitude, velocity gradient. the result is saved in `hist` folder.
- visualize the comparison of PDF (you need to generate `hist` first)
```bash
cd ../
jupyter notebook plot_compare_PDF.ipynb
```
- visualize the comparison of velocity 
```bash
jupyter notebook plot_compare_U_velo.ipynb
```

# Section 3.4: efficient spatial query of 3D HIT dataset
## Data preparation
- it is the same as in Section 3.3

## Training
- MLP, from `width` 36 to 150
```bash
cd compare_inference_speed_memory/MLP

# width = 36
python run_mlp_txyz_siren_uvw.py --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 36 --LAYER_HIDDEN_SPACE 4 --OMEGA_X 30 --NUM_HIDDEN_TIME 10 --LAYER_HIDDEN_TIME 5 --RANK_PARA 5 --NUM_SNAP 20 --BATCH_SIZE 80000
...
# width = 150
python run_mlp_txyz_siren_uvw.py --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 150 --LAYER_HIDDEN_SPACE 4 --OMEGA_X 30 --NUM_HIDDEN_TIME 10 --LAYER_HIDDEN_TIME 5 --RANK_PARA 5 --NUM_SNAP 20 --BATCH_SIZE 80000
```
- NIF, from (spatial) `width` 36 to 150. note that here we consider a low rank of 10 instead of rank 3. also, you should modify `batch_size` according to your computing resources
```bash
cd ../..
cd compare_inference_speed_memory/NIF

# width = 36
python run_nif_txyz_siren_hit_mlp-pn-uvw.py --OMEGA_X 30. --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 36 --LAYER_HIDDEN_SPACE 4 --NUM_HIDDEN_TIME 50 --LAYER_HIDDEN_TIME 2 --BATCH_SIZE 10000 --RANK_PARA 10 --NUM_SNAP 20

# width = 52
python run_nif_txyz_siren_hit_mlp-pn-uvw.py --OMEGA_X 30. --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 52 --LAYER_HIDDEN_SPACE 4 --NUM_HIDDEN_TIME 50 --LAYER_HIDDEN_TIME 2 --BATCH_SIZE 5000 --RANK_PARA 10 --NUM_SNAP 20

# width = 75
python run_nif_txyz_siren_hit_mlp-pn-uvw.py --OMEGA_X 30. --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 75 --LAYER_HIDDEN_SPACE 4 --NUM_HIDDEN_TIME 50 --LAYER_HIDDEN_TIME 2 --BATCH_SIZE 10000 --RANK_PARA 10 --NUM_SNAP 20

# width = 105
python run_nif_txyz_siren_hit_mlp-pn-uvw.py --OMEGA_X 30. --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 105 --LAYER_HIDDEN_SPACE 4 --NUM_HIDDEN_TIME 50 --LAYER_HIDDEN_TIME 2 --BATCH_SIZE 1800 --RANK_PARA 10 --NUM_SNAP 20

# width = 150
python run_nif_txyz_siren_hit_mlp-pn-uvw.py --OMEGA_X 30. --TRAIN_DATA ../../DATA/3dhit-txyz-128-20-uvw.npz --NUM_HIDDEN_SPACE 150 --LAYER_HIDDEN_SPACE 4 --NUM_HIDDEN_TIME 50 --LAYER_HIDDEN_TIME 2 --BATCH_SIZE 700 --RANK_PARA 10 --NUM_SNAP 20
```

## Benchmark the inference time and memory usage

- here we will compare inference time between MLP and NIF when their computational complexity with respect to space is comparable (i.e. width for the MLP = width for the ShapeNet)
- you need to install [fil-profile](https://github.com/pythonspeed/filprofiler for memory profiling)
- now you can run the profiling **OR** you can just checkout the existing logs
```bash
cd ../
bash COMPARE_MEM_MLP.sh
bash COMPARE_MEM_NIF.sh
bash COMPARE_TIME_MLP.sh
bash COMPARE_TIME_NIF.sh
```
- now we can plot the benchmark test on inference time and memory usage. note that you need to manually input the memory usage and inference time in this notebook, which come from (the above fil-profiler pop up windows) by inspecting the memory increment resulting from the neural network model itself, and the network inference time.
```bash
cd ../
jupyter notebook plot_compare_MLP_NIF_inference_memory_time_error.ipynb
```

## Compare against other frameworks over a toy example: 2D video of turbulence. 

We also compared NIF with standard MLP, [SIREN](https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html) and [Fourier Feature Networks](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html). Each row corresponds to a different width size, which approximately determines the computational complexity during inference stage. We found NIF outperforms other frameworks especially when the width size is small (e.g., 36).

## Training

To compare, please go to `./compare_2d_with_variants`, and go to each folder to run the experiments. Detailed command-line argument can be found in those scripts with affix `.sh`. 

## Evaluation

Once the experiments are done, you can evaluate the network performance by running
```bash
bash auto.sh
```
Then a `pred.npz` file will be saved in the corresponding model folder.

### Visualization

Finally, to reproduce the figure 15 and table 2 (in the latest version), go to `plot_compare_2dU_velo.ipynb` for details.