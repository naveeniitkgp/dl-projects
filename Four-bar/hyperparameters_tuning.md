# Hyperparameters tuning for nn_inv_2r model

### Hyperparameters Defaults

| Hyperparameters |     Parser        | Default values |
|:---------------:|:--------------:   |:--------------:|
|   NUM_EPOCHS    | `--num_epochs`    |      500       |
|   BATCH_SIZE    | `--batch_size`    |       32       |
| LEARNING_RATE   |   `--lr`  	      |     1e-3       |
| HIDDEN_DIMS     |   `--hidden_dims` |     [100, 150] |



### Tuning
- python .\Four-bar\src\train.py --results_suffix 12-01-24_v1 --num_epochs 1000 --hidden_dims 100 100 100 100 100

## Modifications


## To Do
- Hyperparameters tuning
- Think about regularization for one to many mapping in deep learning
- Applications - Sythesis (Shigley Book)
- `Given a coupler curve can we find the dimensions of the mechanism`
	+ Think about data generation
