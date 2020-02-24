# MAVEN: Multi-Agent Variational Exploration

## Note

This codebase accompanies paper submission *"MAVEN: Multi-Agent Variational Exploration"*  accepted for NeurIPS 2019.

The codebase is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

### Algorithms

The implementation of the novel **MAVEN** algorithm is done by the authors of the paper. Note that in the codebase MAVEN is referred to by its old name - NoiseMix. The implementation of the **QTRAN**: Learning to factorize with transformation for cooperative multi-agent reinforcement learning algorithm is done by the authors of the paper.

The implementation of the following methods are part of PyMARL:
- **QMIX**: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
- **COMA**: Counterfactual Multi-Agent Policy Gradients
- **VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning
- **IQL**: Independent Q-Learning

### Environments

The implementation of the **Nstep Matrix Game** is done by the authors of the paper.

The authors modified the open-sourced code for StarCraft II from SMAC and added additional scenarios to it.

**The instructions for installing the codebase and running experiments is copied from the original PyMARL repository.**

Tested with Python 3.6.

The zealot_cave maps in the paper are referred to as 3step and 4step respectively in the code.

## Installation instructions

Build the Dockerfile using
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment

```shell
python3 src/main.py --config=noisemix_episode --env-config=sc2 with env_args.map_name=3s5z
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=noisemix_episode --env-config=sc2 with env_args.map_name=3s5z
```

All results will be stored in the `Results` folder.

The hyperparameter configurations for camera ready can be found in the folder plot_keys. 

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep.

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp.

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Citation

Please use the following bibtex entry for citation:
```
@inproceedings{mahajan2019maven,
  title={MAVEN: Multi-Agent Variational Exploration},
  author={Mahajan, Anuj and Rashid, Tabish and Samvelyan, Mikayel and Whiteson, Shimon},
  booktitle={Advances in Neural Information Processing Systems},
  pages={7611--7622},
  year={2019}
}
```
