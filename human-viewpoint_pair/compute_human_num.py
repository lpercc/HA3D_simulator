#compute people number of every house 
import os
import json
import argparse

def compute(pos_data,average_area):
    view_num = len(pos_data)
    #向上取整
    human_num = (view_num*2 // average_area) +1
    return view_num, human_num
     

def main(args):
    GRAPHS = 'connectivity/'
    total_human = 0
    total_viewpoints = 0
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
            scans = [scan.strip() for scan in f.readlines()]
        # 遍历建筑场景列表
    for scan_id in scans:
        with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
            pos_data = json.load(f)
            #print(len(pos_data))
        view_num,human_num = compute(pos_data,67)
        total_human = total_human + human_num
        total_viewpoints = total_viewpoints + view_num
        print(f'house id:{scan_id}, viewpoint number:{view_num}, recommend human number of the house:{human_num}')
    print(f'Total houses:{len(scans)}')
    print(f'Total viewpoints number of these houses:{total_viewpoints}')
    print(f'Total human number of these houses:{total_human}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)