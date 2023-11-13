import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
import json

def main(args):
    npy_path = os.path.join(args.input_dir, 'results.npy')
    assert os.path.exists(npy_path)
    
    with open('../HC-VLN_simulator/human_motion_text.json', 'r') as f:
        human_motion_data = json.load(f)
    num = 0
    for scan_id in human_motion_data:
        for human_view_id in human_motion_data[scan_id]:
            results_dir = os.path.join(args.output_dir, scan_id, human_view_id+'_obj')
            out_npy_path = os.path.join(args.output_dir, scan_id, human_view_id+'_smpl_params.npy')
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            os.makedirs(results_dir)
            
            npy2obj = vis_utils.npy2obj(npy_path, num, 0,
                                        device=args.device, cuda=args.cuda)

            print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
            for frame_i in tqdm(range(npy2obj.real_num_frames)):
                npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

            print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
            npy2obj.save_npy(out_npy_path)
            num = num + 1

        if scan_id == "17DRP5sb8fy":
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    args = parser.parse_args()
    main(args)