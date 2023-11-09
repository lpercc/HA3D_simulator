#!/bin/bash
cd action_generator

for i in {0..9}
do
    formatted_i=$(printf "sample%02d_rep00_obj" $i)
    echo "sample: $formatted_i"
    python -m src.render.rendermdm -i "../samples_humanml_trans_enc_512_000200000_seed10_HC-VLN_tetx_prompts/$formatted_i" -o "../generation/03a8325e3b054e3fad7e1e7091f9d283" -bgi "../03a8325e3b054e3fad7e1e7091f9d283.jpg"
done