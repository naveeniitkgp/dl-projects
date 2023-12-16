# Hyperparameters tuning for nn_inv_2r model

### Hyperparameters Defaults

| Hyperparameters |     Parser     | Default values |
|:---------------:|:--------------:|:--------------:|
|   NUM_EPOCHS    | `--num_epochs` |      100       |
|   BATCH_SIZE    | `--batch_size` |       32       |
| LEARNING_RATE   |   `--lr`  	   |     1e-3       |



### Tuning

| Run | Hyperparameters                           | Date  | Description                            |
|:---:|-------------------------------------------|-------|----------------------------------------|
| 1   | `--batch_size 64`                         | 05/12 | ResNet18 - Validation loss fluctuating.| 



## Modifications
-  Changed the network hidden layers and number of neurons in each layer

## To Do
- Train the model with `lr` 0.001 and `batch_size` of 512


## Papers and Thesis Links
- [Deep learning-based inverse design framework for property targeted novel architectured interpenetrating phase composites](https://doi.org/10.1016/j.compstruct.2023.116783)
	+ Try to understand how these paper has tried to solve the inverse problem.
	+ Look into network architecture and hyperparameters.

