import subprocess


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")


if __name__ == "__main__":
    commands = [
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v1 ",
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v2 --lr 0.0001 "
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v3 --lr 0.0001 --batch_size 64"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v4 --lr 0.0001 --batch_size 64 --zeta_factor 0.5"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v5 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.25"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v6 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.1"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v7 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v8 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05 --num_epochs 750"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v9 --lr 0.0001 --batch_size 64 --zeta_factor 0.5 --zeta 0.05 --num_epochs 1000"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v10 --lr 0.0001 --batch_size 128 --zeta_factor 0.5 --zeta 0.05 --num_epochs 1000"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v11 --lr 0.0001 --batch_size 64 --zeta_factor 0"
        "python .\\src\\nn_inv_2r.py --results_suffix 03-01-24_mod_inv_v11 --lr 0.0001 --batch_size 64 --zeta_factor 0 --num_epochs 1000"
    ]

    for cmd in commands:
        run_command(cmd)
        

