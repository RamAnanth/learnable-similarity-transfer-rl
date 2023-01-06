import numpy as np
import random

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle




import importlib
from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple


from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv


def finite_mdp(env: 'AbstractEnv',
               time_quantization: float = 1.,
               horizon: float = 50.,
               skid_percent: float = 0.) -> object:
    """
    Time-To-Collision (TTC) representation of the state.
    The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
    probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
    duration, under the hypothesis that every vehicles observed will maintain a constant speed (including the
    ego-vehicle) and not change lane (excluding the ego-vehicle).
    For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
    be:
    [0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0]
    The TTC-state is a coordinate (lane, time) within this grid.
    If the ego-vehicle has the ability to change its speed, an additional layer is added to the occupancy grid
    to iterate over the different speed choices available.
    Finally, this state is flattened for compatibility with the FiniteMDPEnv environment.
    :param AbstractEnv env: an environment
    :param time_quantization: the time quantization used in the state representation [s]
    :param horizon: the horizon on which the collisions are predicted [s]
    """
    # Compute TTC grid
    np.random.seed(100)
    random.seed(100)
    grid = compute_ttc_grid(env, time_quantization, horizon)
    # Compute current state
    
    grid_state = (env.vehicle.speed_index, env.vehicle.lane_index[2], 0)
    
    #print("Grid state",env.vehicle.lane_index[2])
    state = np.ravel_multi_index(grid_state, grid.shape)
    
    # Compute transition function
    transition_model_with_grid = partial(transition_model, skid_percent = skid_percent ,grid=grid)

    transition = np.fromfunction(transition_model_with_grid, grid.shape + (env.action_space.n,), dtype=int)
    
    #print(transition)
    transition = np.reshape(transition, (np.size(grid), env.action_space.n))
    
    # Compute reward function
    v, l, t = grid.shape
    lanes = np.arange(l)/max(l - 1, 1)
    speeds = np.arange(v)/max(v - 1, 1)
    
    state_reward = \
        + env.config["collision_reward"] * grid \
        + env.RIGHT_LANE_REWARD * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t)) \
        + env.HIGH_SPEED_REWARD * np.tile(speeds[:, np.newaxis, np.newaxis], (1, l, t))
    
    state_reward = np.ravel(state_reward)
    action_reward = [env.LANE_CHANGE_REWARD, 0, env.LANE_CHANGE_REWARD, 0, 0]
    reward = np.fromfunction(np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
                             (np.size(state_reward), np.size(action_reward)),  dtype=int)

    # Compute terminal states

    collision = grid == 1

    end_of_horizon = np.fromfunction(lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int)
    
    #print(collision, end_of_horizon)
    terminal = np.ravel(collision | end_of_horizon)


    # Creation of a new finite MDP
    try:
        module = importlib.import_module("finite_mdp.mdp")
        mdp = module.DeterministicMDP(transition, reward, terminal, state=state)
        mdp.original_shape = grid.shape
        return mdp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("The finite_mdp module is required for conversion. {}".format(e))


def compute_ttc_grid(env: 'AbstractEnv',
                     time_quantization: float,
                     horizon: float,
                     vehicle: Optional[Vehicle] = None) -> np.ndarray:
    """
    Compute the grid of predicted time-to-collision to each vehicle within the lane
    For each ego-speed and lane.
    :param env: environment
    :param time_quantization: time step of a grid cell
    :param horizon: time horizon of the grid
    :param vehicle: the observer vehicle
    :return: the time-co-collision grid, with axes SPEED x LANES x TIME
    """
    vehicle = vehicle or env.vehicle
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    grid = np.zeros((vehicle.SPEED_COUNT, len(road_lanes), int(horizon / time_quantization)))
    grid_cost = np.zeros((vehicle.SPEED_COUNT, len(road_lanes), int(horizon / time_quantization)))
    for speed_index in range(grid.shape[0]):
        ego_speed = vehicle.index_to_speed(speed_index)
        for other in env.road.vehicles:
            if (other is vehicle) or (ego_speed == other.speed):
                continue
            margin = other.LENGTH / 2 + vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
            for m, cost in collision_points:
                distance = vehicle.lane_distance_to(other) + m
                other_projected_speed = other.speed * np.dot(other.direction, vehicle.direction)
                time_to_collision = distance / utils.not_zero(ego_speed - other_projected_speed)
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(vehicle.lane_index, other.lane_index,
                                                      route=vehicle.route, depth=3):
                    # Same road, or connected road with same number of lanes
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(env.road.network.all_side_lanes(vehicle.lane_index)):
                        lane = [other.lane_index[2]]
                    # Different road of different number of lanes: uncertainty on future lane, use all
                    else:
                        lane = range(grid.shape[1])
                    # Quantize time-to-collision to both upper and lower values
                    for time in [int(time_to_collision / time_quantization),
                                 int(np.ceil(time_to_collision / time_quantization))]:
                        if 0 <= time < grid.shape[2]:
                            # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                            grid[speed_index, lane, time] = np.maximum(grid[speed_index, lane, time], cost)
                            grid_cost[speed_index, lane, time] = cost
    return grid


def transition_model(h: int, i: int, j: int, a: int, grid: np.ndarray, skid_percent: float) -> np.ndarray:
    """
    Deterministic transition from a position in the grid to the next.
    :param h: speed index
    :param i: lane index
    :param j: time index
    :param a: action index
    :param grid: ttc grid specifying the limits of speeds, lanes, time and actions
    """
    # Idle action (1) as default transition
    
        
    next_state = clip_position(h, i, j + 1, grid)
    left = a == 0
    right = a == 2
    faster = (a == 3) & (j == 0)
    slower = (a == 4) & (j == 0)
    
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)
    
    if random.random()<skid_percent:
        print(skid_percent)
        next_state[left] = clip_position(h[left], i[left], j[left], grid)
        next_state[right] = clip_position(h[right], i[right], j[right], grid)
        next_state[slower] = clip_position(h[slower], i[slower], j[slower], grid)
    
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
    
    
    return next_state


def clip_position(h: int, i: int, j: int, grid: np.ndarray) -> np.ndarray:
    """
    Clip a position in the TTC grid, so that it stays within bounds.
    :param h: speed index
    :param i: lane index
    :param j: time index
    :param grid: the ttc grid
    :return: The raveled index of the clipped position
    """
    h = np.clip(h, 0, grid.shape[0] - 1)
    i = np.clip(i, 0, grid.shape[1] - 1)
    j = np.clip(j, 0, grid.shape[2] - 1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)
    return indexes

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.
    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    def __init__(self, config: dict = None ) -> None:
        super(HighwayEnv,self).__init__(config)
        np.random.seed(0)
        self.RIGHT_LANE_REWARD: float = 0.1
        """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

        self.HIGH_SPEED_REWARD: float = 0.4
        """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

        self.LANE_CHANGE_REWARD: float = 0
        """The reward received at each lane change action."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": 0,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": True
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )
        
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


"""
class highway():
    def __init__(self, skid_percent = 0 ):
        
        vehicle_type = 'IDM'
        vehicle2class = {'Linear': 'highway_env.vehicle.behavior.AggressiveVehicle', 'Aggressive' : 'highway_env.vehicle.behavior.AggressiveVehicle', 
        'IDM': 'highway_env.vehicle.behavior.IDMVehicle', 'Defensive': 'highway_env.vehicle.behavior.DefensiveVehicle'}
        
        self.skid_percent = skid_percent
        self.environment = HighwayEnv({'other_vehicles_type': vehicle2class[vehicle_type], 'vehicles_count':200})
        self.environment.reset()
        
        self.mdp = finite_mdp(self.environment, skid_percent = self.skid_percent)
        self.state = self.mdp.state
        self.num_states = self.mdp.transition.shape[0]
        self.num_actions = self.mdp.transition.shape[1]
        
        
    def reset(self):
        
        self.mdp = finite_mdp(self.environment, skid_percent = self.skid_percent)
        self.state = self.mdp.state
        return self.state
        
    def step(self, action):
        reward = self.mdp.reward[self.state, action]
        done = self.mdp.terminal[self.state]
        self.state = self.mdp.transition[self.state, action]
        return self.state, reward, done, {}
"""  
class highway()
if __name__ == '__main__':

    env1 = highway()
    env2 = highway(skid_percent = 0.8)
    
    print(env1.mdp.transition - env2.mdp.transition)


