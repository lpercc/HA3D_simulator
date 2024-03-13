import json
import argparse

# 读取R2R数据
def read_R2R_data(file_path):
    with open(file_path, 'r') as f:
        r2r_data= json.load(f)
    r2r_data_path_num = len(r2r_data)
    print("R2R dataset:")
    print(f"total path:{r2r_data_path_num}")
    print(f"total instructions:{r2r_data_path_num * 3}")
    
    return r2r_data

# 读取HC数据
def read_HC_data(file_path):
    with open(file_path, 'r') as f:
        hc_data= json.load(f)
    hc_data_path_num = len(hc_data)
    print("HC dataset:")
    print(f"total path:{hc_data_path_num}")
    print(f"total instructions:{hc_data_path_num * 3}")
    
    return hc_data, hc_data_path_num

# 获取建筑场景所有视点之间的关系
def read_connection_data(file_path): 
    with open(file_path, 'r') as f:
        connection_data = json.load(f)
    return connection_data

def read_position_data(file_path):
    with open(file_path, 'r') as f:
        position_data = json.load(f)
    return position_data

def save_HC_data(file_path, hc_data):
    with open(file_path, 'w') as f:
        json.dump(hc_data, f, indent=4)

# 获取路径周围的点（可到达的点）
def get_unobstructed_points(path, connection_data):
    path_unobstructed_points = []
    for path_point in path:
        unobstructed_points = connection_data[path_point]['unobstructed']
        for point in unobstructed_points:
            if point not in path_unobstructed_points and point not in path:
                path_unobstructed_points.append(point)
    return path_unobstructed_points

# 获取路径其中一个点的周围的点（可到达的点）
def get_unobstructed_points_(path_point, connection_data):
    path_unobstructed_points_ = connection_data[path_point]['unobstructed']
    return path_unobstructed_points_

# 在path中插入一个点inserted_point，插入位置在pos_point后
def insert_point(pos_point, inserted_point, path):
    # 首先检查pos_point是否存在于path中
    if pos_point in path and inserted_point not in path:
        # 找到pos_point的索引
        index = path.index(pos_point)
        # 在下一个位置插入inserted_point
        path.insert(index + 1, inserted_point)
    else:
        # 如果pos_point不在path中，可能需要抛出错误或者以其他方式处理
        print("The point", pos_point, "is not in the path")
    return path

# 在path中删除一个点pos_point
def delete_point(pos_point, path):
    try:
        # 尝试从path中移除pos_point
        path.remove(pos_point)
    except ValueError:
        # 如果pos_point不在path中，则处理错误
        print("The point", pos_point, "is not in the path")
    return path

def upgrade_data(new_path_data, new_path, new_instructions):
    new_path_data["path"] =  new_path
    new_path_data["instructions"] = new_instructions
    return new_path_data

def main(args):
    hc_data = read_R2R_data(args.hc_data)
    print("Data example:")
    ex_data = hc_data[0]
    keys = list(ex_data.keys())
    for key in keys:
        print("{}:{}".format(key, ex_data[key]))
    scan_id = ex_data["scan"]
    path = ex_data["path"]
    point_of_path = path[0]
    connection_data = read_connection_data('con/con_info/{}_con_info.json'.format(scan_id))
    unobstructed_points = get_unobstructed_points(path, connection_data)
    unobstructed_points_ = get_unobstructed_points_(path[0], connection_data)
    print("Unobstructed points number: {}".format(len(unobstructed_points)))
    print("Path's Unobstructed points {}".format(unobstructed_points))
    print("Point {} Unobstructed points {}".format(point_of_path,unobstructed_points_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--hc_data',default= "./HC-VLN/path.json",help='data file location')
    args = parser.parse_args()
    main(args)