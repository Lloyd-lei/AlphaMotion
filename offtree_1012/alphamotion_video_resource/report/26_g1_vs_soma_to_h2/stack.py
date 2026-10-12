import sys, os, subprocess, numpy as np, imageio.v2 as iio
from PIL import Image
OUT = os.path.dirname(os.path.abspath(__file__))
for key in sys.argv[1:]:
    d = f'{OUT}/{key}'; vids = [f'{d}/g1/review.mp4', f'{d}/soma_h2/review.mp4', f'{d}/g1_h2/review.mp4']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vids[0], '-i', vids[1], '-i', vids[2],
                    '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', f'{d}/triptych.mp4'], check=True)
    rd = iio.get_reader(f'{d}/triptych.mp4'); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int)
    frames = [Image.fromarray(rd.get_data(int(i))).resize((1440, 270)) for i in idx]; rd.close()
    sheet = Image.new('RGB', (1440, 270 * 12), (5, 5, 5))
    for r, im in enumerate(frames): sheet.paste(im, (0, r * 270))
    sheet.save(f'{d}/grid12.jpg', quality=85); print(key, n, 'frames ->', f'{d}/triptych.mp4', f'{d}/grid12.jpg')
