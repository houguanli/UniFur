"""Deliver real alpha WebM + light-checker MP4; existing local FFmpeg only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def encode(root, name):
    source = root/(name+'_transparent.webp')
    transparent = root/(name+'_transparent.webm')
    preview = root/(name+'_checker.mp4')
    assert not transparent.exists() and not preview.exists()
    with Image.open(source) as movie:
        width,height=movie.size; count=movie.n_frames
        base=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgba',
              '-s',f'{width}x{height}','-r','24','-i','pipe:0','-an']
        alpha=subprocess.Popen(base+['-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black@0',
            '-c:v','libvpx-vp9','-pix_fmt','yuva420p','-lossless','1','-auto-alt-ref','0',
            '-deadline','good','-cpu-used','4','-row-mt','1',
            '-threads','4',str(transparent)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        check=subprocess.Popen(base+['-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white',
            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
            '-threads','4',str(preview)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        y,x=np.indices((height,width));gray=np.where((x//24+y//24)%2,238,255).astype(np.uint8)
        background=Image.fromarray(np.repeat(gray[...,None],3,2),'RGB').convert('RGBA')
        try:
            for frame in range(count):
                movie.seek(frame); im=movie.convert('RGBA')
                alpha.stdin.write(im.tobytes())
                bg=background.copy();bg.alpha_composite(im);check.stdin.write(bg.tobytes())
        finally:
            alpha.stdin.close();check.stdin.close()
        for process in [alpha,check]:
            error=process.stderr.read().decode();assert process.wait()==0,error
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-show_streams','-show_format','-of','json',str(transparent)]))
    stream=probe['streams'][0]
    assert stream['tags'].get('alpha_mode')=='1' and int(stream['nb_read_frames'])==count
    # Force the alpha-capable VP9 decoder; the default decoder can discard alpha.
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error',
        '-c:v','libvpx-vp9','-i',str(transparent),'-frames:v','1','-f','rawvideo',
        '-pix_fmt','rgba','pipe:1'])
    alpha_values=np.frombuffer(raw,np.uint8).reshape(-1,4)[:,3]
    assert alpha_values.min()==0 and alpha_values.max()==255
    return dict(source=source.name,frames=count,duration_seconds=float(probe['format']['duration']),
        alpha_video=transparent.name,preview_video=preview.name,alpha_range=[0,255],
        alpha_mode=stream['tags']['alpha_mode'],
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [transparent,preview]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    report={name:encode(args.directory,name) for name in ['full','dark_mane']}
    (args.directory/'VIDEO_RECEIPT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
