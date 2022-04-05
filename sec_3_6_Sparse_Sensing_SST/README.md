# Section 3.6 Sparse reconstruction of sea surface temperature

## Data

- we have prepared the raw datasets in `DATA/`. alternatively, you can download them from [here](https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc) with land masks at [here](https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/lsmask.nc).
- generate the normalized datasets `sst_train_n.npz` for NIF
```bash
cd DATA
python get_normalized_data.py
```

## Run POD-QDEIM 

- to run POD-QDEIM and generate optimal sensor locations at rank 1-10, 20, 50, 100, 200, 300,400,500,600. Just simply run the `qdeim.py` with rank at the argument. 
- for example, to run at rank=1. 
```bash
cd QDEIM 
python qdeim.py 1
```
- then you should see the following `npz` generated
	- `1_sensor_location.npz`
	- `sensing_result_svd_1_sensor_1.npz`

## Run NIF
Note: remember that you need to first run the above POD-QDEIM to determine the best sensor locations, then you can run NIF-SS.

- first, copy the previous sensor_location npz to here
```bash
cd NIF
cp ../QDEIM/*_sensor_location.npz ./
```

- second, we can run the NIF-SS for different number of sensors

	- number of sensor = 5
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 5 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 5
	```

	- number of sensor = 6
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 6 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 6
	```

	- number of sensor = 7
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 7 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 7
	```

	- number of sensor = 8
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 8 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 8
	```

	- number of sensor = 9
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 9 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 9
	```

	- number of sensor = 10
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 10 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 10
	```

	- number of sensor = 20
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 20 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 20
	```

	- number of sensor = 50
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 60 --LAYER_HIDDEN_TIME 2 --RANK_PARA 50 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 50
	```

	- number of sensor = 100
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 100 --LAYER_HIDDEN_TIME 2 --RANK_PARA 100 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 100
	```

	- number of sensor = 200
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 200 --LAYER_HIDDEN_TIME 2 --RANK_PARA 200 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 200
	```

	- number of sensor = 300
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 300 --LAYER_HIDDEN_TIME 2 --RANK_PARA 300 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 300
	```

	- number of sensor = 400
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 400 --LAYER_HIDDEN_TIME 2 --RANK_PARA 400 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 400
	```

	- number of sensor = 500
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 500 --LAYER_HIDDEN_TIME 2 --RANK_PARA 500 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 500
	```

	- number of sensor = 600
	```bash
	python run_nif_txy_siren_sst_autoencoder.py --TRAIN_DATA ../DATA/sst_train_n.npz --NUM_HIDDEN_SPACE 60 --LAYER_HIDDEN_SPACE 2 --NUM_HIDDEN_TIME 600 --LAYER_HIDDEN_TIME 2 --RANK_PARA 600 --NUM_SNAP 832 --BATCH_SIZE 8000 --NUM_SENSOR 600
	```

## Make and save prediction from NIF-SS
```bash
bash all_inference_nif.sh
```

## Reproduce figure 8

```bash
jupyter notebook 0-visualize-sensor-locations.ipynb
cd png-vis
```

## Reproduce top-left of figure 15

```bash
jupyter notebook 1-POD-QDEIM-overfits.ipynb
cd png-error-trend
```

## Reproduce figure 24,25,26
```bash
jupyter notebook 2-compare-nif-ss-with-pod-qdeim.ipynb
cd png-compare-models-sst
cd ../png-error-trend
```
