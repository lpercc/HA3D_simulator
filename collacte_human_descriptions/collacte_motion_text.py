import os
import json

with open('human_motion_text.json', 'r') as f:
    human_motion_data = json.load(f)

with open('human_view_info.json', 'r') as f:
    human_view_data = json.load(f)

GRAPHS = 'connectivity/'
# 每个建筑场景编号
with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
#human_motion_data = {}
for scan_id in scans:
    if scan_id not in human_motion_data:
        human_motion_data[scan_id] = {}
    for view_num in range(len(human_view_data[scan_id])):
        human_view_id = human_view_data[scan_id][view_num]
        human_motion_data[scan_id][human_view_id] = "A man spins around and says hello with his hands."
        

with open('human_motion_text.json', 'w') as f:
    json.dump(human_motion_data, f, indent=4)

with open('HC-VLN_text_prompts.txt', 'w') as f:
    for scan_id in scans:
        if scan_id == "17DRP5sb8fy":
            for human_view_id in human_motion_data[scan_id]:
                f.write(f'{human_motion_data[scan_id][human_view_id]}\n')

            break
        