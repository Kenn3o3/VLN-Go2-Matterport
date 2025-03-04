import pexpect
import os
import time

# Set environment variables
os.environ['DASHSCOPE_API_KEY'] = 'sk-4257a5ed4a2a4bb090dd68df4d10639c'
os.environ['OPENAI_API_KEY'] = 'sk-4257a5ed4a2a4bb090dd68df4d10639c'

# Change to the working directory
os.chdir(os.path.expanduser('~/MCoT-VLN'))

# Define constants
max_attempts = 3        # Maximum retries per episode
wait_time = 10          # Seconds to wait before retrying
activity_timeout = 360   # Seconds to wait for any output
total_timeout = 360     # Total seconds per episode (6 minutes)

# Loop over episode indices from 642 to 1077
for index in range(642, 1078):  # 1078 because range is exclusive
    attempts = 0
    success = False
    
    while attempts < max_attempts and not success:
        attempts += 1
        print(f"Starting attempt {attempts} for episode {index}")
        
        # Construct the command
        command = (
            f"python CLIP-LSTM-Policy/run_pd.py --task=go2_matterport_vision "
            f"--history_length=9 --load_run=2024-09-25_23-22-02 --train "
            f"--episode_index {index}"
        )
        
        # Spawn the process
        child = pexpect.spawn(command)
        
        # Open log file in append mode to keep all attempt outputs
        with open(f"training_collect/episode_{index}.log", "ab") as log_file:
            # Write a separator for this attempt
            log_file.write(f"\n--- Starting attempt {attempts} for episode {index} ---\n".encode())
            child.logfile = log_file  # Log output to file
            
            # Monitor the simulation
            start_time = time.time()
            while time.time() - start_time < total_timeout:
                try:
                    # Expect stop message or any output line
                    index_match = child.expect(
                        [
                            "\[\d+\.\d+s\] Simulation is stopped\. The app will keep running "
                            r"with physics disabled\. Press Ctrl\+C or close the window to exit the app\."
                        ],
                        timeout=activity_timeout
                    )
                    if index_match == 0:
                        # Stop message found
                        success = True
                        break
                    # If index_match == 1, some output was received, continue looping
                except pexpect.TIMEOUT:
                    print(f"No output in {activity_timeout} seconds for episode {index}, attempt {attempts}")
                    child.kill(9)  # Terminate due to inactivity
                    child.wait()
                    break
                except pexpect.EOF:
                    print(f"EOF for episode {index}, attempt {attempts}. Process crashed.")
                    break
            
            # Handle post-loop actions
            if success:
                # Simulation completed, terminate cleanly
                child.sendcontrol('c')
                child.wait()
            elif child.isalive():
                # Total timeout reached without completion
                print(f"Episode {index} did not complete within {total_timeout} seconds, terminating")
                child.kill(9)
                child.wait()
        
        # Handle retry logic
        if not success:
            if attempts < max_attempts:
                print(f"Retrying episode {index} after {wait_time} seconds")
                time.sleep(wait_time)
            else:
                print(f"Failed to complete episode {index} after {max_attempts} attempts. Skipping.")
        else:
            print(f"Successfully completed episode {index}")