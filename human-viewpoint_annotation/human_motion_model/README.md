# Human motion models generation
## generate human motion skeleton
get MDM repo from https://github.com/GuyTevet/motion-diffusion-model
```bash
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model
```
follow the MDM repo README, after step 3 (https://github.com/GuyTevet/motion-diffusion-model#3-download-the-pretrained-models)
run the script
```bash
#generate from human motion text prompts
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ../HA-VLN_human_motion_prompts.txt --output_dir $HA3D_SIMULATOR_DTAT_PATH/samples_humanml_trans_enc_512_000200000_seed10 --num_repetitions 3 --batch_size 145
```
You may also define:
  --device id.
  --seed to sample different prompts.
  --motion_length (text-to-motion only) in seconds (maximum is 9.8[sec]).

## Render SMPL mesh
```bash
cp ../skeleton2smpl.py ./visualize
python -m visualize.skeleton2smpl --input_dir $HA3D_SIMULATOR_DTAT_PATH/samples_humanml_trans_enc_512_000200000_seed10 --output_dir $HA3D_SIMULATOR_DTAT_PATH/human_motion_meshes
```