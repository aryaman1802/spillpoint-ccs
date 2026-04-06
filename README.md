# spillpoint-ccs

![Spillpint-gif](seed31_apr5.gif)

To generate the above gif, run the following:

```python
from spillenv2 import MultiRegionCO2StorageEnv

seed = 31
env = MultiRegionCO2StorageEnv(fixed_seed=seed, obs_mode='timelapse')
actions = [9]*1200 + [3]*5000
infos = env.create_gif(actions, save_path=f"give_it_some_name.gif", fps=32, show_wells=[1,2])
```