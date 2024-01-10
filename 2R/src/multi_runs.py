import subprocess


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")


if __name__ == "__main__":
    commands = [
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v1 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v3 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v4 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v5 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v6 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v7 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v8 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v9 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v10 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20, 20",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v11 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50, 50",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v12 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100, 100",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v13--lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20, 20, 20",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v14 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50, 50, 50",
        "python .\\src\\nn_inv_2r.py --results_suffix 06-01-24_mod_inv_v15 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100, 100, 100",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v1_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v2_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v3_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v4_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v5_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v6_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v7_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v8_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v9_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v10_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20, 20",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v11_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50, 50",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v12_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100, 100",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v13_2--lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 20, 20, 20, 20, 20",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v14_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 50, 50, 50, 50, 50",
        "python .\\src\\nn_inv_2r_2.py --results_suffix 06-01-24_mod_inv_v15_2 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000 --hidden_dims 100, 100, 100, 100, 100",

    ]

    for cmd in commands:
        run_command(cmd)
        

