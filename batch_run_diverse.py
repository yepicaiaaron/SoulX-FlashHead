import os
import subprocess
import re

audios = [
    'test1_rap.wav',
    'test2_auctioneer.wav',
    'test3_laugh.wav',
    'test4_shouting.wav',
    'test5_opera.wav'
]

# Set face_ratio to 3.5
def patch_facecrop(ratio):
    with open('flash_head/utils/facecrop.py', 'r') as file:
        content = file.read()
    content = re.sub(r'face_ratio=[\d\.]+,', f'face_ratio={ratio},', content)
    with open('flash_head/utils/facecrop.py', 'w') as file:
        file.write(content)

patch_facecrop(3.5)

for audio in audios:
    name = audio.split('.')[0]
    outfile = f'benchmark/results/aaron_{name}_pro.mp4'
    if not os.path.exists(outfile):
        cmd = f'python3 generate_video.py --ckpt_dir ./models/SoulX-FlashHead-1_3B --wav2vec_dir ./models/wav2vec2-base-960h --model_type pro --use_face_crop True --cond_image benchmark/aaron.jpg --audio_path benchmark/{audio} --save_file {outfile}'
        print(f'Running: {cmd}')
        subprocess.run(cmd, shell=True, check=True)
