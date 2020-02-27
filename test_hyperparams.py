#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
import numpy as np
import json
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import SAC
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import time


# In[3]:


# Central agent controlling one of the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ['Building_3']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']


# In[4]:


params = {"non_shiftable_load": False, "month": True, "solar_gen": True, "t_out_pred_12h": False,  "t_out_pred_6h": True}
for p_key in params:
    print("Testing Parameter %s" % (p_key))
    p_val = params[p_key]
    with open(building_state_actions, "r") as f:
        data = json.load(f)
    if data[building_ids[0]]["states"][p_key] == p_val:
        print("Skipping parameter with same value")
        continue
    data[building_ids[0]]["states"][p_key] = p_val
    with open(building_state_actions, "w") as f:
        json.dump(data, f, indent=4)

    env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 1)
    average = 0
    for t in range(10):
        model = SAC(MlpPolicy_SAC, env, verbose=0, learning_rate=0.01, gamma=0.99, tau=3e-4, batch_size=2048, learning_starts=8759)
        start = time.time()
        model.learn(total_timesteps=8760*7, log_interval=1000)
        print("Time: %f" % (time.time()-start))

        obs = env.reset()
        dones = False
        counter = np.empty(8760)
        i = 0
        while dones==False:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            counter[i] = rewards
            i += 1
        average += (np.sum(counter) - average)/(t+1)
        env.cost()
        print("")
    print("Average: %f" % (average))

    data[building_ids[0]]["states"][p_key] = not p_val
    with open(building_state_actions, "w") as f:
        json.dump(data, f, indent=4)
    


# In[5]:


# Plotting winter operation
interval = range(0,8759)
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using SAC for storage(kW)'])
