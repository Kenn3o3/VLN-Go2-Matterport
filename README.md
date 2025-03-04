# Dataset Description

## Overview

This dataset is designed for **training and evaluating models on Vision Language Navigation task**. It is collected using unitree go2 navigating through indoor environments from the **Matterport dataset** (provided by VLN-CE-Isaac benchmark), rendered with **Isaac Lab**. Each episode in the dataset represents a navigation task where the robot follows an **expert path** using a **Proportional-Derivative (PD) controller**, capturing **visual observations** (RGB images and depth maps) and **discrete actions** at regular intervals (40 simulation steps). The dataset is intended to support research in areas such as imitation learning, reinforcement learning, and vision-language navigation.

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

### Data Collection Frequency
Data (RGB images, depth maps, and actions) is collected every 20 (`k_steps`) simulation steps, where `k_steps` is defined by the `--action_repeat` argument (default is 20) in the `run_pd.py --train` code. This ensures that the dataset captures key decision points during the navigation task rather than every simulation step to reduce space.

---

## Data Collection Process

The dataset is generated in **training mode** using the `--train` flag in the `run_pd.py` script. The process involves a simulated robot navigating through environments sourced from the **Matterport dataset**, which consists of scanned real-world indoor spaces (e.g., houses, apartments). The simulation is powered by **Omniverse Isaac Sim**, providing realistic visual and spatial data.

### Episode Setup
Each episode is configured based on metadata from the `vln_ce_isaac_v1.json.gz` file in the `VLN-CE-Isaac`, which includes:
- **Start Position**: The initial position and orientation of the robot.
- **Goal Position**: The target location the robot must reach.
- **Instruction**: A natural language description of the task (e.g., "Navigate to the bedroom").
- **Expert Path**: A sequence of ground-truth positions representing an optimal or near-optimal trajectory to the goal.

### Navigation with PD Controller
In training mode, actions are selected using a **Proportional-Derivative (PD) controller** that follows the expert path. The controller operates as follows:
1. **Robot State**: The robot’s current position and orientation are obtained from the simulation.
2. **Target Selection**: The controller identifies the closest point on the expert path and selects a look-ahead point (default `look_ahead = 2` steps ahead).
3. **Yaw Calculation**: The desired yaw (rotation) is computed based on the relative position of the look-ahead point in the robot’s body frame.
4. **Action Selection**:
   - If the desired yaw rate exceeds a threshold (`w_threshold = 0.05 rad/s`), the robot selects:
     - `"Turn left"` (angular velocity = 0.5 rad/s) if the yaw rate is positive.
     - `"Turn right"` (angular velocity = -0.5 rad/s) if the yaw rate is negative.
   - Otherwise, the robot selects `"Move forward"` (linear velocity = 0.6 m/s).
5. **Execution**: The selected action is executed for `k_steps` simulation steps before collecting data and selecting the next action.

### Episode Termination
An episode ends when one of the following conditions is met:
- **Success**: The robot reaches within 1 meter of the goal position.
- **Failure**: The maximum number of steps (default `max_steps = 4000`) is reached.
- **Other Termination**: The environment signals completion for other reasons (e.g., collision or timeout).

### Data Logging
At each data collection step (every `k_steps`):
- The **RGB image** is captured from the robot’s camera.
- The **depth map** is recorded to provide spatial information.
- The **action** chosen by the PD controller is logged.
- The instruction remains constant and is saved once per episode.

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
  - **RGB Images**: Saved in their original resolution as PNG files. For visualization in the web UI, they are resized to 256x256 pixels, but the dataset retains the full-resolution images.
  - **Depth Maps**: Saved as NumPy arrays for compatibility with machine learning frameworks like PyTorch or TensorFlow.

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
