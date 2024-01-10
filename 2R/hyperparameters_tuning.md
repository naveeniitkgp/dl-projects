# Hyperparameters tuning for nn_inv_2r model

### Hyperparameters Defaults

| Hyperparameters |     Parser        | Default values |
|:---------------:|:--------------:   |:--------------:|
|   NUM_EPOCHS    | `--num_epochs`    |      500       |
|   BATCH_SIZE    | `--batch_size`    |       32       |
| LEARNING_RATE   |   `--lr`  	      |     1e-3       |
| EPOCH_ZETA      |   `--zeta_factor` |        0.2     |
| ZETA            |   `--zeta`  	  |     0.5        |





### Tuning
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v2 --lr 0.0001 "
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v3 --lr 0.0001 --batch_size 64"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v4 --lr 0.0001 --batch_size 64 --zeta_factor 0.5"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v5 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.25"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v6 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.1"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v7 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v8 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05 --num_epochs 750"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v9 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05 --num_epochs 1000"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v10 --lr 0.0001 --batch_size 128 --zeta_factor 0.5 --zeta 0.05 --num_epochs 1000"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v11 --lr 0.0001 --batch_size 64 --zeta_factor 0"
- "python .\\src\\nn_inv_2r.py --results_suffix 21-12-23_mod_inv_v11 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000"

## Modifications
-  

## To Do

### 2R
- Plot for each test cases for
	+ `Done` $\theta_1 \in \left( 0, 2\pi \right)$ and $\theta_2 \in \left( 0, 2\pi \right)$
	+ `Analysis left` $\theta_1 \in \left( -\pi, \pi \right)$ and $\theta_2 \in \left( -\pi, \pi  \right)$
- Modify the code
	+ `Done` add $\LaTeX$ codes for the plots in the main code.
	+ `Not required` add a code that saves the last saved `.pth` file separetly in a `.txt` file so that it will be easy to call in code rather than looking it everytime.
	+ `Not required` Include early stopping (to avoid overfitting)
- Try with smaller size of layers and neurons.
	+ `Done`User defined layers and  number of neuraons in each layers
- There are more points towards the edges - check distribution of points
	+ Try with generating $\left( r, \theta \right) \rightarrow \left( X, Y \right)$

### Fourbar
- Think of generating the data
	+ Predict $\phi$ given $\theta_1$ $\rightarrow$ Freudenstine equation.
	+ Predict $(x,y)$ given $\theta_1$ $\rightarrow$ Coupler curve.

### PUMA 560
- $^0\mathbf{O}_6 = \left( x, y, z \right)^T$ is a function of $\theta_1$, $\theta_2$, $\theta_3$ and D-H constants
	+ Get a single equation in $\theta_2$, $x$, $y$, $z$ and D-H constants -- Underlying equation.
## Papers and Thesis Links
- [Deep learning-based inverse design framework for property targeted novel architectured interpenetrating phase composites](https://doi.org/10.1016/j.compstruct.2023.116783)
	+ Try to understand how this paper has attempted to solve the inverse problem.
	+ Look into network architecture and hyperparameters. 
- [Machine learning for (meta-)materials design](https://github.com/mmc-group/ML-for-materials-design) - Tutorials
- [On solving the inverse kinematics problem using neural networks](https://doi.org/10.1109/M2VIP.2017.8211457)
- [2R IK using neighbourhood information](https://github.com/OmarJItani/Deep-Neural-Network-for-Solving-the-Inverse-Kinematics-Problem)
- [Local_INN: Implicit Map Representation and Localization with Invertible Neural Networks](https://doi.org/10.48550/arXiv.2209.11925)
	+ Suggested my Krishna, a student of Jihsnu Keshavan.
	+ Supposably it can do inverse problems using invertible NN.
- Inverse-designed spinodoid metamaterials
	+ [Paper](https://doi.org/10.1038/s41524-020-0341-6)
	+ [GitHub](https://github.com/mmc-group/inverse-designed-spinodoids)

## Intresting things!
- [Draw NN](http://alexlenail.me/NN-SVG/AlexNet.html)

## Analysis

| Name                           | $\theta_{1min}$ | $\theta_{1max}$ | $\theta_{2min}$ | $\theta_{2max}$ | Loss   | Notes                           |
|:------------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|:------:|:-------------------------------:|
| `ik_nn_03-01-24_mod_inv_v1`    | 1.3             | 7.9             | 0.2             | 3               | 1e-1   | -                               |
| `ik_nn_03-01-24_mod_inv_v2`    | 0.3             | 6.9             | 0.4             | 3               | 2e-1   | -                               |
| `ik_nn_03-01-24_mod_inv_v3`    | -1              | 5               | 0.2             | 3               | 0.7e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v4`    | -0.5            | 6               | 0               | 3               | 1e-1   | -                               |
| `ik_nn_03-01-24_mod_inv_v5`    | -0.5            | 6               | 0               | 3               | 0.7e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v6`    | -0.3            | 6               | -3              | -0.7            | 0.7e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v7`    | -2              | 5               | 0.1             | 3               | 0.7e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v8`    | -0.2            | 6               | 0.3             | 3               | 0.5e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v9`    | 0.5             | 7               | 0.3             | 2.8             | 0.4e-1 | -                               |
| `ik_nn_03-01-24_mod_inv_v10`   | -0.7            | 5.2             | 0.2             | 3               | 0.1e-1 | There is a spike like extension |
| `ik_nn_03-01-24_mod_inv_v11`   | -4              | 2               | -3              | -0.2            | 0.2e-1 | -                               |