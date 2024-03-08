import os
import json
import argparse
from render.rendermdm import HE_fusion 

## pre render video
def main(args):

    global src_view, tar_view

    # human_view_info.json每个建筑场景的人物视点的信息（视点编号）
    # 一共90个建筑场景数据
    with open('human_motion_text.json', 'r') as f:
        human_view_data = json.load(f)

    GRAPHS = 'connectivity/'
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
            scans = [scan.strip() for scan in f.readlines()]

    video_list = []
    record_dict = {}
    
    ## get skybox number
    # 每个建筑场景中的视点视角朝向
    with open("con/heading_info.json", 'r') as f:
        heading_data = json.load(f)
    
    # 获取建筑场景列表
    if args.mode == 'run_all':
        scan_list = scans
    else:
        scan_list = []
        scan_list.append(args.scan)
    
    # 遍历建筑场景列表
    for scan_id in scan_list:
    #scan_id = "B6ByNegPMKs"
        print(scan_id)
        # 获取该建筑场景的全景视图存放路径
        view_path = os.path.join(args.input_dir, "data/v1/scans/{}/matterport_panorama_images".format(scan_id))
        # 输出路径
        output_path = os.path.join(args.output_dir, "data/v1/scans/{}/matterport_panorama_video".format(scan_id))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 获取建筑场景所有视点信息（视点之间的关系）
        with open('con/con_info/{}_con_info.json'.format(scan_id), 'r') as f:
            connection_data = json.load(f)
            #print(len(connection_data))

        with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
            pos_data = json.load(f)
            #print(len(pos_data))

        record_dict[scan_id] = {}


        # 遍历建筑场景中每个人物视点，即人物所在位置的视点
        for human_view_id in human_view_data[scan_id]:
            # 人物视点编号
            human_motion = human_view_data[scan_id][human_view_id][0]
            human_model_id = human_view_data[scan_id][human_view_id][1]
            human_heading = human_view_data[scan_id][human_view_id][2]
            print(f"Human motion: {human_motion}")
            #print(human_view_id)
            # 随机选择人物动作编号
            # action_select = random.randrange(0, 39)
            
            # 遍历所有视点
            for num, val in connection_data.items():
                try:
                    # 判断该视点是否可见目标视点（人物）
                    if human_view_id == num:
                        val["visible"].append(num)
                        print(f"human_view_id:{num}")
                    if human_view_id in val['visible']:
                        # 源视点编号
                        agent_view_id = num
                        info_list = []
                        motion_path = os.path.join(args.motion_dir, human_motion.replace(' ', '_').replace('/', '_'), f"{human_model_id}_obj")
                        #print(motion_path)
                        bgd_img_path = os.path.join(view_path, agent_view_id+'.jpg')
                        #print(bgd_img_path)
                        output_video_path = os.path.join(output_path, agent_view_id+".mp4")
                        #print(output_video_path)
                        agent_loc = [pos_data[agent_view_id][0], pos_data[agent_view_id][1], pos_data[agent_view_id][2]]
                        #print(agent_loc)
                        human_loc = [pos_data[human_view_id][0], pos_data[human_view_id][1], pos_data[human_view_id][2]]
                        try:
                            agent_heading = heading_data[scan_id][agent_view_id][0]
                        except KeyError:
                            agent_heading = 180
                        info_list.append(agent_loc) 
                        info_list.append(agent_heading)
                        info_list.append(human_loc)
                        info_list.append(human_motion)
                        if not os.path.exists(output_video_path) or human_view_id==agent_view_id:
                            HE_fusion(motion_path, output_video_path, bgd_img_path, agent_view_id, agent_loc, human_loc, agent_heading, human_heading,scan_id,human_view_id)
                        else:
                            print(f"{agent_view_id} has already fusioned")
                        video_list.append(output_video_path)
                        try:
                            record_dict[scan_id][agent_view_id] += (info_list)
                        except:
                            record_dict[scan_id][agent_view_id] = []
                            record_dict[scan_id][agent_view_id] +=(info_list)
                except KeyError:
                    print(human_view_id)
                    #break
    
    with open('video.txt', 'w') as f:
        f.write('\n'.join(video_list))
    # with open('add_info.json', 'w') as info_file:
    #     json.dump(record_dict, info_file, indent = 3)
        
if __name__ == "__main__":
    basic_data_dir = os.getenv('VLN_DATA_DIR')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='run_all')
    parser.add_argument('--scan', default='17DRP5sb8fy')
    parser.add_argument('--input_dir', default=basic_data_dir)
    parser.add_argument('--output_dir', default=basic_data_dir)
    parser.add_argument('--motion_dir', default=os.path.join(basic_data_dir,"human_motion_meshes"))
    args = parser.parse_args()
    main(args)