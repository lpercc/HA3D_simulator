from multiprocessing import Process
import os
import argparse

def run_program(command, suppress_output=False):
    if suppress_output:
        command += " >/dev/null 2>&1"
    print(command)
    os.system(f'python {command}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pn', default=1)
    args = parser.parse_args()
    viewpoint_s = 0
    viewpoint_e = 0
    viewpoint_count = 10567
    per = viewpoint_count // int(args.pn)
    print(f"{int(args.pn)} Process,{per} viewpoints per process")
    for i in range(int(args.pn)):
        viewpoint_s = per * i
        suppress_output = i+1 != int(args.pn)
        if i+1 == int(args.pn):
            viewpoint_e = viewpoint_count
        else:
            viewpoint_e = per * (i+1)

        Process(target=run_program, args=(f"scripts/extract_video_features.py --gpu 0 --viewpoint_s {viewpoint_s} --viewpoint_e {viewpoint_e}", suppress_output)).start()