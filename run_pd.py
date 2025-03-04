# scripts/VLMNavAgent/VLMAgent_run.py
from web_ui.app import WebUI
import argparse
from queue import Queue
from models.clip_lstm_policy import CLIPLSTMPolicy
import os
import time
import math
import gzip, json
from datetime import datetime
from omni.isaac.lab.app import AppLauncher
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from log_manager import LogManager
from image_history_manager import ImageHistoryManager
from inference_manager import InferenceManager
import torch
import cv2
import numpy as np
from typing import Dict, Any
import cli_args
import torch.nn.functional as F
from PIL import Image

# Add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes to run.")
parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--train", action="store_true", default=False, help="Run in training mode with user input.")
parser.add_argument("--action_repeat", default=20, type=int, help="Number of simulation steps to repeat each action.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--draw_pointcloud", action="store_true", default=1, help="Draw pointcloud.")
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
import gymnasium as gym
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
import omni.isaac.lab.utils.math as math_utils
from scipy.spatial.transform import Rotation
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper

def quat2eulers(q0, q1, q2, q3):
    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)
    return roll, pitch, yaw

class Planner:
    def __init__(self, env, env_cfg, args_cli, simulation_app):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.simulation_app = simulation_app
        self.k_steps = args_cli.action_repeat  # Number of steps to repeat each action
        self.action_queue = Queue()  # Queue for user actions (optional for monitoring)
        self.web_ui = WebUI(action_queue=self.action_queue)
        self.log_manager = LogManager(log_dir="logs/experiments", web_ui=self.web_ui)
        self.image_history_manager = ImageHistoryManager(n_history=2)
        self.step_counter = 0
        self.max_steps = 3200
        self.action_map = {
            "Move forward": (0.6, 0.0, 0.0),
            "Turn left": (0.0, 0.0, 0.5),
            "Turn right": (0.0, 0.0, -0.5)
        }
        self.action_to_idx = {"Move forward": 0, "Turn left": 1, "Turn right": 2}
        self.idx_to_action = {0: "Move forward", 1: "Turn left", 2: "Turn right"}
        self.action_velocities = [
            (0.6, 0.0, 0.0),  # Move forward
            (0.0, 0.0, 0.5),  # Turn left
            (0.0, 0.0, -0.5)  # Turn right
        ]
        # PD controller parameters
        self.look_ahead = 2
        self.Kp_yaw = 0.5 / np.pi  # Scale yaw to range [-0.5, 0.5]
        self.w_threshold = 0.05     # Threshold for turning (rad/s)

        # Expert path visualization (unchanged)
        # self.marker_cfg = CUBOID_MARKER_CFG.copy()
        # self.marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
        # self.marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        # self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.unwrapped.device).repeat(1, 1)
        # for i in range(self.env_cfg.expert_path_length):
        #     expert_point_visualizer = VisualizationMarkers(self.marker_cfg)
        #     expert_point_visualizer.set_visibility(True)
        #     point = np.array(self.env_cfg.expert_path[i]).reshape(1, 3)
        #     default_scale = expert_point_visualizer.cfg.markers["cuboid"].scale
        #     larger_scale = 2.0 * torch.tensor(default_scale, device=self.env.unwrapped.device).repeat(1, 1)
        #     expert_point_visualizer.visualize(point, self.identity_quat, larger_scale)

        # Initialize and load the policy model (unchanged)
        self.policy = CLIPLSTMPolicy(device=self.env.unwrapped.device)
        model_path = "saved_model/policy.pth"
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path))
            print("Loaded saved model from", model_path)
        else:
            print("No saved model found at", model_path, ". Using random policy.")

    def select_action_with_pd(self):
        """Select a discrete action using a PD controller based on the expert path."""
        # Get robot's current position and orientation
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()

        # Compute rotation matrix from quaternion
        rot = Rotation.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])  # [x, y, z, w]
        rot_matrix = rot.as_matrix()

        # Find closest point on expert path
        distances = np.linalg.norm(self.env_cfg.expert_path[:, :2] - robot_pos[:2], axis=1)
        closest_idx = np.argmin(distances)

        # Target a point ahead
        target_idx = min(closest_idx + self.look_ahead, len(self.env_cfg.expert_path) - 1)
        target_pos = self.env_cfg.expert_path[target_idx]

        # Compute relative position in world frame
        relative_pos = target_pos - robot_pos

        # Transform to robot's body frame
        relative_pos_body = rot_matrix.T @ relative_pos

        # Compute desired yaw
        desired_yaw = np.arctan2(relative_pos_body[1], relative_pos_body[0])

        # Compute desired yaw rate (P control)
        ω_d = self.Kp_yaw * desired_yaw

        # Map to discrete action
        if abs(ω_d) > self.w_threshold:
            action = "Turn left" if ω_d > 0 else "Turn right"
        else:
            action = "Move forward"

        return action

    def run_episode(self, episode_idx: int) -> None:
        obs, infos = self.env.reset()
        current_instruction = self.env_cfg.instruction_text
        self.log_manager.start_episode(episode_idx, current_instruction)
        
        # **Initialize episode_finished to False at the start**
        self.web_ui.current_data['episode_finished'] = False
        self.web_ui.update_data(self.web_ui.current_data)

        # Training data buffers
        instructions = []
        rgbs = []
        depths = []
        actions = []
        
        self.step_counter = 0
        self.robot_path = []
        success = False
        done = False
        
        print("Started episode. Visualization at http://localhost:5000")

        if self.args_cli.train:
            # Training mode: Use PD controller to select actions
            while self.step_counter < self.max_steps and not done:
                # Select action every k_steps
                if self.step_counter % self.k_steps == 0:
                    action = self.select_action_with_pd()
                    vel_command = torch.tensor(self.action_map[action], device=self.env.unwrapped.device)
                
                # Execute action and collect data
                obs, _, done, infos = self.env.step(vel_command.clone())
                self.step_counter += 1
                
                # Update state
                robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
                depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
                bgr_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                goal_distance = np.linalg.norm(robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
                self.robot_path.append((robot_pos[0], robot_pos[1]))
                
                # Collect training data
                if self.step_counter % self.k_steps == 0:
                    rgbs.append(rgb_image_np.copy())
                    depths.append(depth_map.copy())
                    actions.append(action)
                
                # Update UI (optional monitoring)
                resized_rgb = cv2.resize(rgb_image_np, (256, 256))
                resized_depth = cv2.resize(depth_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                self.image_history_manager.add_image(resized_rgb, resized_depth, 0, action, robot_pos)
                self.log_manager.log_step(
                    self.step_counter, self.robot_path, bgr_image_np, depth_map, action,
                    goal_distance, self.max_steps - self.step_counter
                )
                self.web_ui.current_data.update({
                    'vlm_prompt': current_instruction,
                    'vlm_response': action,
                    'waiting_for_action': False
                })
                self.web_ui.update_data(self.web_ui.current_data)
                
                # Check if goal is reached
                if goal_distance < 1.0:
                    success = True
                    done = True
                    break
        else:
            # Inference mode: Use the policy to predict actions (unchanged)
            hidden = None
            robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
            depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
            while self.step_counter < self.max_steps and not done:
                action_logits, new_hidden = self.policy(
                    [current_instruction], [rgb_image_np], [depth_map[None, :, :]], hidden
                )
                action_idx = torch.argmax(action_logits, dim=1).item()
                action = self.idx_to_action[action_idx]
                vel_command = torch.tensor(self.action_velocities[action_idx], device=self.env.unwrapped.device)
                for _ in range(self.k_steps):
                    if self.step_counter >= self.max_steps:
                        break
                    obs, _, done, infos = self.env.step(vel_command.clone())
                    self.step_counter += 1
                    robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                    rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
                    depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
                    bgr_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                    goal_distance = np.linalg.norm(robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
                    self.robot_path.append((robot_pos[0], robot_pos[1]))
                    _, new_hidden = self.policy(
                        [current_instruction], [rgb_image_np], [depth_map[None, :, :]], new_hidden
                    )
                    resized_rgb = cv2.resize(rgb_image_np, (256, 256))
                    resized_depth = cv2.resize(depth_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                    self.image_history_manager.add_image(resized_rgb, resized_depth, 0, action, robot_pos)
                    self.log_manager.log_step(
                        self.step_counter, self.robot_path, bgr_image_np, depth_map, action,
                        goal_distance, self.max_steps - self.step_counter
                    )
                    self.web_ui.current_data.update({
                        'vlm_prompt': current_instruction,
                        'vlm_response': action,
                        'waiting_for_action': False
                    })
                    self.web_ui.update_data(self.web_ui.current_data)
                    if goal_distance < 1.0:
                        success = True
                        done = True
                        break
                hidden = new_hidden

        # Episode outcome
        status = "Success: Reached the goal within 1 meter" if success else "Failure: Did not reach the goal" if done else "Failure: Reached maximum steps"
        self.log_manager.set_status(status)
        print(status)

        # **Set episode_finished to True after episode completes**
        self.web_ui.current_data['episode_finished'] = True
        self.web_ui.update_data(self.web_ui.current_data)

        # Save training data
        if self.args_cli.train and success:
            print("saving data")
            data_dir = f"training_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_scene_{self.env_cfg.scene_id}_episode_{self.args_cli.episode_index}"
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "instructions.txt"), "w") as f:
                f.write(current_instruction + "\n")
            with open(os.path.join(data_dir, "actions.txt"), "w") as f:
                for action in actions:
                    f.write(action + "\n")
            rgb_dir = os.path.join(data_dir, "rgbs")
            os.makedirs(rgb_dir, exist_ok=True)
            for t, rgb in enumerate(rgbs):
                Image.fromarray(rgb).save(os.path.join(rgb_dir, f"rgb_{t}.png"))
            depth_dir = os.path.join(data_dir, "depths")
            os.makedirs(depth_dir, exist_ok=True)
            for t, depth in enumerate(depths):
                np.save(os.path.join(depth_dir, f"depth_{t}.npy"), depth)
        if self.args_cli.train and not success:
            print("not success, not saving data")

    def start_loop(self):
        """Start running multiple episodes."""
        if self.args_cli.train:
            print("Running in training mode with automated PD controller.")
        self.run_episode(self.args_cli.episode_index)
        time.sleep(3)

if __name__ == "__main__":
    # Environment Configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    vel_command = torch.tensor([0, 0, 0])
    episode_idx = args_cli.episode_index 
    dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    with gzip.open(dataset_file_name, "rt") as f:
        deserialized = json.loads(f.read())
        episode = deserialized["episodes"][episode_idx]
        if "go2" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.4)
        elif "h1" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+1.0)
        else:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.5)
        env_cfg.scene.disk_1.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+2.5)
        env_cfg.scene.disk_2.init_state.pos = (episode["reference_path"][-1][0], episode["reference_path"][-1][1], episode["reference_path"][-1][2]+2.5)
        wxyz_rot = episode["start_rotation"]
        init_rot = wxyz_rot
        env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
        env_cfg.goals = episode["goals"]
        env_cfg.episode_id = episode["episode_id"]
        env_cfg.scene_id = episode["scene_id"].split('/')[1]
        env_cfg.traj_id = episode["trajectory_id"]
        env_cfg.instruction_text = episode["instruction"]["instruction_text"]
        env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
        env_cfg.reference_path = np.array(episode["reference_path"])
        expert_locations = np.array(episode["gt_locations"])
        env_cfg.expert_path = expert_locations
        env_cfg.expert_path_length = len(env_cfg.expert_path)
        env_cfg.expert_time = np.arange(env_cfg.expert_path_length) * 1.0
    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")  

    print("scene_id: ", env_cfg.scene_id)
    print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Low Level Policy
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.join(os.path.dirname(__file__), "../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(env, low_level_policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)

    planner = Planner(env, env_cfg, args_cli, simulation_app)
    planner.start_loop()
    simulation_app.close()
    print("closed!!!")