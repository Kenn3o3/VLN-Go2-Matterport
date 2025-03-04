import os

main_folder = 'training_data'

subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

present_episodes = []
for folder in subfolders:
    if 'episode_' in folder:
        parts = folder.split('_')
        episode_number = parts[-1]
        try:
            episode_number = int(episode_number)
            present_episodes.append(episode_number)
        except ValueError:
            pass

all_episodes = set(range(0, 1078))

present_set = set(present_episodes)
missing_episodes = all_episodes - present_set

missing_episodes_list = sorted(list(missing_episodes))

with open('missing_episodes.txt', 'w') as f:
    for episode in missing_episodes_list:
        f.write(f"{episode}\n")