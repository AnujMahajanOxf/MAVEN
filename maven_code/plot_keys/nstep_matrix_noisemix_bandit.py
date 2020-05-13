from run_experiment import extend_param_dicts

server_list = [
    ("gandalf", [1,2,3,4,5,6,7], 2),
]

label = "10step_matrix__bandit_rnn__1ep__15_May_2019"

config = "noise_qmix_parallel"
env_config = "nmatrix"

n_repeat = 20 # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 100 * 1000 + 5 * 1000,

    "env_args.steps": 10,
    "env_args.good_branches": 2,

    "batch_size_run": 1,

    "test_interval": 1000,
    "test_nepisode": 64,
    "test_greedy": True,
    "log_interval": 1000,
    "runner_log_interval": 2000,
    "learner_log_interval": 2000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
    "buffer_size": 1000,
    "epsilon_finish": 0.01,
    "epsilon_anneal_time": 100,

    "discrim_size": 32,
}

name = "noisemix"
extend_param_dicts(param_dicts, shared_params,
    {
        "name": name,
        "noise_dim": [16],
        "bandit_iters": 100,
        "noise_bandit": [True],
        "rnn_discrim": [True, False],
        "mi_loss": [0.01, 0.1, 1],
        # "entropy_scaling": [0.001, 0.01, 0.1]
    },
    repeats=parallel_repeat)
