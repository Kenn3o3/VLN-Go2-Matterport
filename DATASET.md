# Dataset Description

## Overview

This dataset is designed for **training and evaluating models on Vision Language Navigation task**. It is collected using unitree go2 navigating through indoor environments from the **Matterport .usd dataset** (provided by VLN-CE-Isaac benchmark), rendered with **Isaac Lab**. Each episode in the dataset represents a navigation task where the robot follows an **expert path** using a **Proportional-Derivative (PD) controller**, capturing **visual observations** (RGB images and depth maps) and **discrete actions** at regular intervals (20 simulation steps). The dataset is intended to support research in areas such as imitation learning, reinforcement learning, and vision-language navigation.

---

## Dataset Structure

The dataset is organized into per-episode directories, each containing data collected during a single navigation task. The directory name follows the format `training_data/YYYY-MM-DD_HH-MM-SS_scene_{scene_id}_episode_{episode_index}`, where:
- `YYYY-MM-DD_HH-MM-SS` is the timestamp of data collection.
- `scene_id` identifies the Matterport scene (e.g., a specific house or building).
- `episode_index` is the index of the episode within the dataset.

Inside each episode directory, the following files and directories are present:

- **`instruction.txt`**  
  A text file containing the **natural language instruction** for the episode (e.g., "Go to the kitchen and find the red chair"). This instruction remains constant throughout the episode and defines the navigation task.

- **`actions.txt`**  
  A text file listing the **discrete actions** taken by the robot at each data collection step. Each line corresponds to an action, such as:
  - `"Move forward"`
  - `"Turn left"`
  - `"Turn right"`

- **`rgbs/`**  
  A directory containing **RGB images** captured from the robot's camera at each data collection step. These are saved as PNG files, named sequentially (e.g., `rgb_0.png`, `rgb_1.png`, etc.).

- **`depths/`**  
  A directory containing **depth maps** corresponding to the RGB images. These are saved as NumPy arrays (e.g., `depth_0.npy`, `depth_1.npy`, etc.) for efficient loading and processing.

## Data Collection Pipeline (To be updated)

---

## Usage

This dataset is a valuable resource for various research and development tasks, including:

- **Imitation Learning**  
  Train models to replicate the PD controller’s behavior by learning mappings from instructions and visual observations (RGB images and depth maps) to discrete actions.

- **Reinforcement Learning**  
  Use the dataset as a demonstration set for offline reinforcement learning or as a starting point for online RL agents to refine navigation policies.

- **Vision-Language Navigation**  
  Develop models that interpret natural language instructions and navigate visual environments, leveraging the paired instructions, RGB images, depth maps, and actions.

- **Evaluation**  
  Benchmark navigation agents by comparing their trajectories or action sequences to the expert-guided demonstrations in the dataset.

---

## Additional Information

- **Instructions** (Collected from VLN-CE-Isaac) 
  Instructions are sourced from the Matterport dataset’s `vln_ce_isaac_v1.json.gz` file and are designed to reflect real-world navigation tasks (e.g., "Find the sofa in the living room").

- **Environment**  (Collected from VLN-CE-Isaac)
  The environments are USD files from the Matterport dataset, representing real-world indoor spaces with detailed geometry and textures, providing a challenging and realistic setting for navigation.

- **Preprocessing**  
  - **RGB Images**: Saved in their original resolution as PNG files.
  - **Depth Maps**: Saved as NumPy arrays.

- **Action Space**  
  We discretized the actions available to the robot as:
  - `"Move forward"`: Linear velocity of 0.6 m/s forward.
  - `"Turn left"`: Angular velocity of 0.5 rad/s to the left.
  - `"Turn right"`: Angular velocity of -0.5 rad/s to the right.
  You may adjust the code as need.

- **Expert Path**  
  The expert path is derived from ground-truth locations in the Matterport dataset, providing a reference trajectory that the PD controller aims to follow.

- **Simulation Parameters** (`run_pd.py`)
  - **Action Repeat**: Controlled by `--action_repeat` (default 20), determining how many simulation steps each action persists.
  - **Maximum Steps**: Set to 4000 by default, adjustable via code modification.
  - **PD Controller Tuning**: Parameters like `Kp_yaw = 0.5 / π`, `w_threshold = 0.05`, and `look_ahead = 2` can be adjusted to influence action selection.
