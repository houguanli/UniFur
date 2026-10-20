"""Side-by-side existing native RGBA frames; no exposure/alpha edits."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'a025_all_fur_wind_20261020'))
import encode_video


def main(root):
    old=root.parent/'a025_curve_frame_wind_20261020/geometry_v2'
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',28)
    result={}
    for mode in ('gust','steady'):
        out=root/'run001'/mode
        with Image.open(old/mode/'fringe_transparent.webp') as before,Image.open(out/'fringe_transparent.webp') as after:
            assert before.n_frames==after.n_frames==96 and before.size==after.size
            frames=[];w,h=before.size
            for i in range(96):
                before.seek(i);after.seek(i)
                frame=Image.new('RGBA',(2*w,h+48));draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,2*w,47),fill=(255,255,255,255))
                draw.text((18,7),'BEFORE: nearby Strand carrier',font=font,fill='black')
                draw.text((w+18,7),'AFTER: matched Shell carrier',font=font,fill='black')
                frame.alpha_composite(before.convert('RGBA'),(0,48))
                frame.alpha_composite(after.convert('RGBA'),(w,48))
                frames.append(frame)
            source=out/'before_after_fringe_transparent.webp';assert not source.exists()
            frames[0].save(source,save_all=True,append_images=frames[1:],duration=round(1000/24),loop=0,lossless=True)
            frames[14].save(out/'before_after_fringe_frame014_transparent.png')
        result[mode]=encode_video.encode(out,'before_after_fringe')
    (root/'COMPARISON_VIDEO_RECEIPT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
