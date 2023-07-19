from stable_baselines import GAIL
from env_1joint_norew_targetasobs import ExcaRobo ############## GANTI ENVIRONMENT DISINI ##########
import matplotlib.pyplot as plt

#for plotting
theta2_array = []
theta3_array = []
theta4_array = []
theta2target_array = []
theta3target_array = []
theta4target_array = []
theta2dot_array = []
theta3dot_array = []
theta4dot_array = []
if __name__ == '__main__': 
  SIM_ON = 1
### LOAD AND RENDER ###
  model = GAIL.load("1joint_norew_targetasobs")
  env = ExcaRobo(SIM_ON)
  obs = env.reset()
#   while True:
for i in range(1300):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    observation_theta2, observation_thetadot2, target = obs
    theta2_array.append(observation_theta2)
    theta2target_array.append(target)
    theta2dot_array.append(observation_thetadot2)
    env.render()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
# Plot theta 2
axes[0].plot(theta2_array, label='theta 2')
axes[0].plot(theta2target_array, label='theta 2 target')
axes[0].set_xlabel('i')
axes[0].set_ylabel('theta (rad)')
axes[0].set_title('Plot Theta 2')
axes[0].set_ylim(-1.57,1.57)
axes[0].legend()
axes[1].plot(theta2dot_array, label='theta 2 dot')
axes[1].set_xlabel('i')
axes[1].set_ylabel('theta dot')
axes[1].set_title('Plot Theta 2 Dot')
axes[1].legend()
plt.suptitle('Plot Theta, Theta Dot, dan Theta Target pada model GAIL')
plt.tight_layout()
plt.show()