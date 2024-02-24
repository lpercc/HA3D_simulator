import json
import os
def get_human_info(basic_data_dir, scan_id, agent_view_id):
    motion_dir = os.path.join(basic_data_dir,"human_motion_meshes")
        # 一共90个建筑场景数据
    with open('human_motion_text.json', 'r') as f:
        human_view_data = json.load(f)
        # 遍历建筑场景中每个人物视点，即人物所在位置的视点
            # 获取建筑场景所有视点信息（视点之间的关系）
    with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
        pos_data = json.load(f)
        #print(len(pos_data))
    with open('con/con_info/{}_con_info.json'.format(scan_id), 'r') as f:
        connection_data = json.load(f)
    for human_view_id in human_view_data[scan_id]:
        # 人物视点编号
        human_motion = human_view_data[scan_id][human_view_id][0]
        human_model_id = human_view_data[scan_id][human_view_id][1]
        human_heading = human_view_data[scan_id][human_view_id][2]
        try:
            # 判断该视点是否可见目标视点（人物）
            if human_view_id == agent_view_id:
                connection_data[agent_view_id]["visible"].append(agent_view_id)
                print(f"human_view_id:{agent_view_id}")
            if human_view_id in connection_data[agent_view_id]['visible']:
                motion_path = os.path.join(motion_dir, human_motion.replace(' ', '_').replace('/', '_'), f"{human_model_id}_obj")
                human_loc = [pos_data[human_view_id][0], pos_data[human_view_id][1], pos_data[human_view_id][2]]
                return human_heading, human_loc, motion_path
        except KeyError:
            pass
    return None, None, None