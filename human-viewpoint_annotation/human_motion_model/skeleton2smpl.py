import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
import json

MODEL_NUM = 3

def main(args):
    npy_path = os.path.join(args.input_dir, 'results.npy')
    assert os.path.exists(npy_path)
    
    with open(os.path.join(args.input_dir, 'results.txt'), 'r') as f:
        human_motion = f.readlines()
    hm_texts = [s.replace('\n', '') for s in human_motion]

    for hm_text in tqdm(hm_texts, desc="Human motion skeleton to SMPL model"):
        for i in range(MODEL_NUM):
            results_dir = os.path.join(args.output_dir, hm_text, str(i)+'_obj')
            out_npy_path = os.path.join(args.output_dir, hm_text, str(i)+'_smpl_params.npy')
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            os.makedirs(results_dir)
            motion_num = hm_texts.index(hm_text)
            
            rep_num = i
            print(f"Motion text:{hm_text} rep number:{rep_num}")
            npy2obj = vis_utils.npy2obj(npy_path, motion_num, rep_num,
                                        device=args.device, cuda=args.cuda)

            print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
            for frame_i in tqdm(range(npy2obj.real_num_frames)):
                npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

            print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
            npy2obj.save_npy(out_npy_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_HA-VLN_human_motion_prompts", help='stick figure mp4 file to be rendered.')
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getenv('HA3D_SIMULATOR_DTAT_PATH'),"human_motion_meshes"))
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    args = parser.parse_args()
    main(args)