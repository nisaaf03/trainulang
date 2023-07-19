import gym
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data
import random

"NGECOPY DARI 'ENV_1JOINT_NOREW_TARGETASOBS_V2' BEDANYA: "
" nilai theta targetnya beda (jadi 0,85)"

class ExcaRobo(gym.Env):
    def __init__(self, sim_active):
        super(ExcaRobo, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
            physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 3_000
        self.dt = 1.0/240.0
        self.max_theta = [1.03] #, 1.51, 3.14]    
        self.min_theta = [-0.954] #, -0.1214, -0.32]
        self.joints_targets = np.array([[-0.7], #,1.4,0], # posisi joint awal kah? buat target reset
                                        [-0.144], #,0.59,1.47],
                                        [-0.257], #,1.17,1.19],
                                        [-0.294], #,1.437,1.823],
                                        [-0.444], #,1.458,1.859],
                                        [-0.344]]) #,1.46,0.276]])
        self.position_targets = np.array([[8.27,0,4.48],
                                          [9.817,0,0.942],
                                          [8.22,0,0.75],
                                          [6.64,0,1.104],
                                          [6.64,0,2.08],
                                          [8.19,0,2.11]])
        self.orientation_targets =  np.array([-0.7, 
                                              -0.5, #-1.841,
                                              -0.85, #2
                                              0.33, #0.966, #3
                                              1.05 #0.74, #4
                                              -1.292])
        
        self.orientation1joint = np.array([self.orientation_targets[2]])
        self.idx_target = 0
        self.n_target = len(self.orientation_targets)
        self.observation_space = spaces.Box(low =-np.inf, high = np.inf, shape= (3,), dtype=np.float32)
        self.action_space = spaces.Box(low = -0.3, high = 0.3, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Discrete(3)
        self.steps_left = np.copy(self.MAX_EPISODE)
        self.start_simulation()

    def step(self, action):
        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action, force= 250_000)
        # p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action[1], force= 250_000)
        # p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action[2], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation Error
        self.theta_now, self.theta_dot_now = self._get_joint_state()
        self.orientation_now = self.normalize(-sum(self.theta_now))
        # print('orientasi = ',self.orientation_now)

        orientation_error = self.rotmat2theta(
            self.rot_mat(self.orientation_target)@self.rot_mat(self.orientation_now).T
        )
        desired_orientation_velocity = 0.9*orientation_error

        self.orientation_velocity = (self.orientation_now-self.orientation_last)/self.dt
        #Position error

        # posisi dan linear velocity buat ngitung rewardnya aja?

        self.position_now, self.link_velocity = self._get_link_state()

        vec = np.array(self.position_now) - self.position_target
        desired_linear_velocity = -0.9*vec

        # reward_dist = 4*np.exp(-np.linalg.norm(desired_linear_velocity-self.link_velocity))
        # reward_orientation = -0.02*(desired_orientation_velocity-self.orientation_velocity)**2
        # reward_ctrl = -0.0075*np.linalg.norm(action)

        # reward = reward_dist + reward_ctrl + reward_orientation

        reward = 0

        self.new_obs = self._get_obs() #action = action, 
                                    #  desired_orientation_velocity = desired_orientation_velocity, 
                                    #  desired_linear_velocity = desired_linear_velocity, 
                                    #  error = vec, 
                                    #  orientation_error = orientation_error)
        error_theta = self.theta_now - self.orientation1joint

        if np.any(self.theta_now > np.array(self.max_theta)) or np.any(self.theta_now < np.array(self.min_theta)) or abs(error_theta) <= 0.05 :
            done = True
            # punishment = -1000
            punishment = 0
            self.reward = reward+punishment
        else:
            done = bool(self.steps_left<0)
            self.steps_left -= 1
            self.reward = reward
        #Update State
        self.orientation_last = self.orientation_now
        self.last_act = action
        self.cur_done = done
        # print('### ONE STEP DONE ###')
        return self.new_obs, self.reward, done, {}

    def start_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        print('start simulation')

        ## Setup Physics
        p.setGravity(0,0,-9.8)

        ## Load Plane
        planeId = p.loadURDF("plane.urdf")

        ## Load Robot
        startPos = [0,0,1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
        # Get the random index of targets
        self.idx_target = random.randint(1,self.n_target-1)
        self.position_target, self.orientation_target = self.position_targets[self.idx_target], self.orientation_targets[self.idx_target]

        #Reset Simulation
        p.resetSimulation()
        self.start_simulation()
        idx_start = self.idx_target+1
        start_position = self.joints_targets[idx_start] if idx_start<=5 else self.joints_targets[0]
        # try:
        #     idx_start = random.randint(0,self.n_target)
        #     start_position = self.joints_targets[idx_start]
        # except IndexError:
        #     start_position = np.array([0,0,0])
        vel = np.zeros(1)

        while True:
            theta_now, _ = self._get_joint_state()
            for i in range(1):
                err = self.rotmat2theta(self.rot_mat(start_position[i])@self.rot_mat(theta_now[i]).T)
                vel[i] = 2*err

            p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = vel[0], force= 150_000)
            # p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = vel[1], force= 150_000)
            # p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = vel[2], force= 150_000)

            #Update Simulations
            p.stepSimulation()
            # print('proses reset....')
            time.sleep(self.dt)

            if np.all(abs(theta_now-start_position)<1e-1):
                p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
                # p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
                # p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
                p.stepSimulation()
                time.sleep(self.dt)
                break

        #Get Joint State
        self.theta_now, self.theta_dot_now = self._get_joint_state()
        self.orientation_last = self.normalize(-sum(self.theta_now))
        self.orientation_now = self.normalize(-sum(self.theta_now))
        self.orientation_velocity = (self.orientation_now-self.orientation_last)/self.dt

        #Get Link State
        self.position_now, self.link_velocity = self._get_link_state()

        self.steps_left = np.copy(self.MAX_EPISODE)
        self.last_act = np.array([0])
        self.cur_done = False
        self.new_obs = self._get_obs() #action = self.last_act, 
                                    #  desired_orientation_velocity = 0, 
                                    #  desired_linear_velocity = np.array([0,0,0]), 
                                    #  error = np.array([0,0,0]), 
                                    #  orientation_error = 0)
        return self.new_obs

    def render(self, mode='human'):
        print(f'State {self.new_obs}, action: {self.last_act}, done: {self.cur_done}')

    def _get_joint_state(self): #MENDAPATKAN NILA THETA (dan theta dot) DARI 3 JOINT
        theta0, theta1, theta2 = p.getJointStates(self.boxId, [2,3,4])
        theta_now = self.normalize(np.array([theta0[0]])) #, theta1[0], theta2[0]]))
        theta_dot_now = np.array([theta0[1]]) #, theta1[1], theta2[1]])
        return theta_now, theta_dot_now

    def normalize(self, x): #NORMALISASI THETA AGAR DIANTARA -PI HINGGA PI
        return ((x+np.pi)%(2*np.pi)) - np.pi

    def _get_obs(self): #self, action, desired_orientation_velocity, desired_linear_velocity, error, orientation_error):
        # theta2_now = self.theta_now[0]
        # thetadot2_now = self.theta_dot_now[0]
        return np.concatenate(
            [
                self.theta_now,
                self.theta_dot_now,
                self.orientation1joint
                # action,
                # desired_linear_velocity,
                # error,
                # self.position_now,
                # self.link_velocity,
                # [self.orientation_now, self.orientation_velocity, desired_orientation_velocity, orientation_error]
            ]
        )

    def rot_mat(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    
    def rotmat2theta(self, matrix):
        return np.arctan2(matrix[0,2],matrix[0,0])

    def _get_link_state(self): # ngubah jadi dapetin posisi link arm pertama (index nya 2)
        (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(self.boxId, 2, computeLinkVelocity=1, computeForwardKinematics=1)
        
        return linkWorldPosition, worldLinkLinearVelocity
    