"""
Extend OpenAI gym Reacher-v2 implementation to include different friction values to design transfer learning scenario
"""
import numpy as np
from gym import utils
import tempfile
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.mujoco_env import *
from gym import spaces
import gym

class MujocoEnvDiscrete(gym.Env):
    """
    Modifed Superclass to be used for all MuJoCo environment.
    """
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        self.action_space = spaces.Discrete(9)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        
XML = """
<mujoco model="reacher">
	<compiler angle="radian" inertiafromgeom="true"/>
	<default>
		<joint armature="1" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
	</default>
	<option gravity="%f 0 -9.81" integrator="RK4" timestep="0.01"/>
	<worldbody>
		<!-- Arena -->
		<geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
		<geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<!-- Arm -->
		<geom conaffinity="0" contype="1" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
		<body name="body0" pos="0 0 .01">
			<geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
			<joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
			<body name="body1" pos="0.1 0 0">
				<joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0"  type="hinge"/>
				<geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
				<body name="fingertip" pos="0.11 0 0">
					<geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
				</body>
			</body>
		</body>
		<!-- Target -->
		<body name="target" pos=".1 -.1 .01">
			<joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
			<joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
			<geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
		</body>
	</worldbody>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
	</actuator>
</mujoco>"""

#Obstacle: <!-- Obstacle --><geom name="obstacle" type="box" pos="0 -0.05 %f" size="0.01 0.01 0.01" rgba="0.2 0.5 0.2 1" conaffinity="1"/>

class ReacherDiscreteEnv(MujocoEnvDiscrete, utils.EzPickle):
    def __init__(self, friction = -0.1):
        # Possible actions are maximum torque or minimum torque or zero torque for each actuator
        # make the action lookup from integer to real action
        self.name = 'Reacher'
        actions = [-1., 0., 1.]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)
        
        self.max_episode_steps = 50 # v2
        self._elapsed_steps = 0
        self._tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".xml")
        self._tempfile.write(XML % friction)
        self._tempfile.flush()
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, self._tempfile.name, 2)
        
    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward = reward_dist
        action = self.action_dict[a]
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self._elapsed_steps = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

if __name__ == '__main__':
    env = ReacherDiscreteEnv(friction = 0.02)
    num_episodes = 50
    print('Observation Space',env.observation_space.shape[0])
    print('Action Dimension',env.action_space.n)
    
    
    for i in range(num_episodes):
        observation = env.reset()
        episode_length = 0
        episode_return = 0
        while True:
            action = np.random.randint(9)
            episode_length = episode_length + 1
            env.render()
            observation,reward,done, _ = env.step(action)
            episode_return = episode_return + reward
            if done:
                print('Episode Return', episode_return)
                break
    