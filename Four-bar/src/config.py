import torch
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size: Default: 32')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of Epochs: Default: 500')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate: Default: 0.001')
parser.add_argument('--zeta_factor', type=float, default=0.2,
                    help='percentage of epochs till zeta is used: Default: 0.2')
parser.add_argument('--zeta', type=float, default=0.5, help='Zeta value: Default: 0.5')
#
parser.add_argument('--results_suffix', type=str, default='#######', help='Suffix for results folder at each run')
parser.add_argument('--hidden_dims', nargs='+', type=int,
                    default=[100, 150], help='List of the number of neurons in each hidden layer. Default: 100 150')
parser.add_argument("--print_summary", action="store_true",
                    help="Print the summary of the model: Default: False")

args = parser.parse_args()


NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCH_ZETA = args.zeta_factor * NUM_EPOCHS
ZETA = args.zeta
Print_Model_Summary = args.print_summary

# Parameters for the model
input_dim_freud = 1
output_dim_freud = 1
hidden_dims_freud = args.hidden_dims
