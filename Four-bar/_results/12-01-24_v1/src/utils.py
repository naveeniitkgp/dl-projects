# Other imports
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modules
from config import parse_arg
args = parse_arg()


def save_metrics(metrics, filename='metrics.txt'):
    with open(filename, 'w') as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")


# def calculate_r2_scores(actuals, predictions, target_names):
#     for i in range(len(target_names)):
#         score = r2_score(actuals[:, i], predictions[:, i])
#         print(f'R² score for {target_names[i]}: {score:.4f}')


def calculate_r2_scores(actuals, predictions, target_names, folder_path):
    # Construct the file path
    file_path = f"{folder_path}/r_squared_scores.md"

    # Open the file and write the scores
    with open(file_path, 'w') as file:
        for i in range(len(target_names)):
            score = r2_score(actuals[:, i], predictions[:, i])
            print(f'R² score for {target_names[i]}: {score:.4f}')
            file.write(f'R$^2$ score for {target_names[i]}: {score:.4f}\n')


def generate_parity_plots(actuals, predictions, target_names, results_folder):
    for i, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5, label='Test Points')
        min_val, max_val = min(actuals[:, i].min(), predictions[:, i].min()), max(
            actuals[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Parity Plot for {target_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_folder, f'parity_plot_{target_name}.png'), dpi=300)
        plt.close()


def save_hyperparameters_md(file_path):
    # Convert hidden_dims list to a string for display
    hidden_dims_str = ', '.join(map(str, args.hidden_dims))

    # Create a Markdown table with hyperparameters and their values
    markdown_content = f"""
| Hyperparameters       | Values          |
|:---------------------:|:---------------:|
| NUM_EPOCHS            | {args.num_epochs} |
| BATCH_SIZE            | {args.batch_size} |
| LEARNING_RATE         | {args.lr}        |
| EPOCH_ZETA            | {args.zeta_factor} |
| ZETA                  | {args.zeta}      |
| RESULT_FOLDER_SUFFIX  | {args.results_suffix} |
| HIDDEN_LAYER_DIMS     | [{hidden_dims_str}] |
"""

    # Write the Markdown content to the specified file
    with open(file_path, 'w') as file:
        file.write(markdown_content)
