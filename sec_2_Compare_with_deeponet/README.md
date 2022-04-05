## Example: a simple travelling wave - comparing with DeepONet

1. generating data
```bash
python gen_tw_data.py
```

2. training
- run DeepONet
```bash
cd DEEPONET-16
python train_datafit_surrogate.py --TRAIN_DATA ../tw_train.npz --NETWORK_TYPE DEEPONET --RANK 16 --N_S 30 --N_T 30 --ACT swish --L_R 1e-3 --BATCH_SIZE 256 --EPOCH 800001
```
- run NIF
```bash
cd NIF-small
python train_datafit_surrogate.py --TRAIN_DATA ../tw_train.npz --NETWORK_TYPE NIF --RANK 0 --N_S 2 --N_T 2 --ACT swish --L_R 1e-3 --BATCH_SIZE 2048 --EPOCH 800001
```

3. visualize the result

```bash
jupyter notebook compare_with_deeponet.ipynb
```