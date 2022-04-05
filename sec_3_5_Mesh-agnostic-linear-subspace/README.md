# Section 3.5 Modal analysis on adaptive mesh data

## Data: flow over a cylinder 

This classical flow over a cylinder problem has been sovled using an AMR solver, `PeleLM`. We have prepared the pointwise datasets in `DATA/cylinder_uv_area_siren_n_eco.npz`. Note that this data contains area, which is used to obtain the area-weighted integral. 

## Training

Here we train low-rank NIF on the cylinder flow problem.

```bash
python run_nif_siren_lowrank_cyd_vpod_area.py --TRAIN_DATA ./DATA/cylinder_uv_area_siren_n_eco.npz --DIM_U 2 --NUM_HIDDEN_SPACE 300 --LAYER_HIDDEN_SPACE 3 --NUM_HIDDEN_TIME 100 --LAYER_HIDDEN_TIME 3 --RANK 10 --BATCH_SIZE 2000 --RUN_ID 1
```

## Modal analysis with DMD
```bash
jupyter notebook plot_cyd_koopman.ipynb
```
