"""Make a GIF/contact sheet of captured camera frames without loading Isaac Sim."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
p = argparse.ArgumentParser()
p.add_argument('--episode', required=True, type=Path)
a = p.parse_args()
paths = sorted((a.episode/'rgb').glob('*.png'))
if not paths:
    raise SystemExit('No frames')
frames = []
for pth in paths[::2]:
    with Image.open(pth) as im:
        frames.append(im.convert('RGB').copy())
frames[0].save(a.episode/'preview.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
sheet = Image.new('RGB', (4*256, 3*280), 'white')
draw = ImageDraw.Draw(sheet)
for j in range(12):
    idx = round(j*(len(paths)-1)/11)
    with Image.open(paths[idx]) as im:
        sheet.paste(im.resize((256, 256)), ((j%4)*256, (j//4)*280))
    draw.text(((j%4)*256+5, (j//4)*280+258), f'frame {idx}', fill='black')
sheet.save(a.episode/'contact_sheet.jpg')
print(a.episode/'preview.gif')
