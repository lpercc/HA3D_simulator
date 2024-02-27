import json
import os
import math

def get_human_info(basic_data_dir, scan_id, agent_view_id):
    motion_dir = os.path.join(basic_data_dir,"human_motion_meshes")
        # 一共90个建筑场景数据
    with open('human_motion_text.json', 'r') as f:
        human_view_data = json.load(f)
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


def get_human_on_path():

    with open('human_motion_text.json', 'r') as f:
        human_view_data = json.load(f)
    human_count = 0
    for scan in human_view_data:
        human_count = human_count + len(scan)
    print(f"human count:{human_count}")
    r2r_data = read_R2R_data("path.json")

    new_r2r_data = []

    num = 0


    Beginning_num = 0
    Obstacle_num = 0
    Around_num = 0
    End_num = 0
    on_path_num = 0

    for r2r_data_item in r2r_data:
        human_info = []
        scan_id = r2r_data_item["scan"]
        path = r2r_data_item["path"]
        path_id = r2r_data_item["path_id"]
        with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
            pos_data = json.load(f)
            #print(len(pos_data))
        with open('con/con_info/{}_con_info.json'.format(scan_id), 'r') as f:
            connection_data = json.load(f)
    
        path_visible_points = get_visible_points(path, connection_data)

        for visible_point in path_visible_points:
            if visible_point in human_view_data[scan_id]:
                human_rel_pos = get_rel_pos(visible_point, path, path_id, pos_data)
                human_info.append({
					"human_viewpoint":visible_point,
					"human_rel_pos":human_rel_pos,
					"human_description":human_view_data[scan_id][visible_point][0]
				})
        if len(human_info) > 0:
            num += 1
        r2r_data_item["human"] = human_info
        new_r2r_data.append(r2r_data_item)
        for item in human_info:
            value = item["human_rel_pos"]
            #print(value)
            if value in ["Beginning", "Obstacle", "End"]:
                on_path_num += 1
                break
    print(f"{num} / {len(new_r2r_data)}")
    print(f"paths with human:{on_path_num}")
    #print(f"Beginning_num:{Beginning_num}, Obstacle_num:{Obstacle_num}, End_num:{End_num}, Around_num:{Around_num}, None_num:{None_num}")
    with open("new_r2r_data.json", 'w') as f:
        json.dump(new_r2r_data, f, indent=4)
    #return

def get_rel_pos(human_point, path, path_id, pos_data):
    loc_dsc = [
        "Beginning",
        "Obstacle", 
        "Around",
        "End"
    ]
    min_distance = 1000
    for index, path_point in enumerate(path):
        distance = compute_distance(human_point, path_point, pos_data)
        if distance < min_distance:
            min_distance = distance
            if distance < 1.5 and index == 0:
                human_rel_pos=loc_dsc[0]
            elif distance < 1.5 and index == len(path)-1:
                human_rel_pos=loc_dsc[-1]
            elif distance < 1.5:
                human_rel_pos=loc_dsc[1]
            else:
                human_rel_pos=loc_dsc[2]
    return human_rel_pos

def load_viewpointids():
    GRAPHS = "connectivity/"
    viewpointIds = []
    with open(GRAPHS + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    print("Loaded %d viewpoints" % len(viewpointIds))
    return viewpointIds

def compute_distance(viewpointId1, viewpointId2, pos_data):
    x_dis = pos_data[viewpointId1][0] - pos_data[viewpointId2][0]
    y_dis = pos_data[viewpointId1][1] - pos_data[viewpointId2][1]
    z_dis = pos_data[viewpointId1][2] - pos_data[viewpointId2][2]
    squared_sum = x_dis**2 + y_dis**2 + z_dis**2
    return math.sqrt(squared_sum)


# 读取R2R数据
def read_R2R_data(file_path):
    with open(file_path, 'r') as f:
        r2r_data= json.load(f)
    r2r_data_path_num = len(r2r_data)
    print("R2R dataset:")
    print(f"total path:{r2r_data_path_num}")
    print(f"total instructions:{r2r_data_path_num * 3}")
    
    return r2r_data


# 获取路径周围可见的点（包括路径点本身）
def get_visible_points(path, connection_data):
    path_visible_points = []
    try:
        for path_point in path:
            visible_points = connection_data[path_point]['visible']
            for point in visible_points:
                if point not in path_visible_points:
                    path_visible_points.append(point)
    except KeyError:
        print(connection_data[path_point])
    return path_visible_points

 def count_points_seen_human():
    GRAPHS = 'connectivity/'
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]

    viewpoints_counts = 0
    human_visible_counts = 0
    for scan_id in scans:
        with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
            pos_data = json.load(f)
        viewpoints_counts = len(pos_data) + viewpoints_counts
        for viewpoint in pos_data:
            human_heading, human_loc, motion_path = get_human_info("./", scan_id, viewpoint)
            if human_heading is not None:
                human_visible_counts += 1
    print(f"human visible points {human_visible_counts} / All points {viewpoints_counts}")
   

if __name__ == '__main__':
    count_points_seen_human()