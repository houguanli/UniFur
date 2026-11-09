"""Check an A025 replay's frames, encoded videos and solver diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageChops


def command(args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def mane_crop(root, config):
    camera = config['camera']
    if (camera['width'], camera['height']) != (1200, 800):
        raise ValueError('The mane crop uses the approved 1200x800 camera')
    crop = root / 'mane_comparison.mp4'
    if crop.exists():
        return
    left = 'Matched rod reference' if config['reference'] != 'Reduced-mode control' else 'Reduced-mode control'
    right = f"Rods {config['version']}"
    filters = ("[0:v]split=2[a][b];[a]crop=520:520:280:180,scale=780:780[a1];"
               "[b]crop=520:520:1480:180,scale=780:780[b1];[a1][b1]hstack,"
               "pad=1560:820:0:40:color=0x161c24,"
               f"drawtext=text='{left}':x=15:y=12:fontcolor=white:fontsize=20,"
               f"drawtext=text='{right}':x=795:y=12:fontcolor=0x78f0be:fontsize=20[out]")
    command(['ffmpeg', '-v', 'error', '-n', '-i', str(root / 'comparison/preview.mp4'),
             '-filter_complex', filters, '-map', '[out]', '-c:v', 'libx264', '-crf', '18',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(crop)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--against', type=Path, help='optional archived run for exact optimized PNG pixel comparison')
    parser.add_argument('--mane-crop', action='store_true')
    args = parser.parse_args()
    root = args.output
    report = json.loads((root / 'report.json').read_text())
    config = report['config']
    if args.mane_crop:
        mane_crop(root, config)
    paths = [root / name / 'preview.mp4' for name in ('optimized', 'comparison', 'diagnostic')]
    if (root / 'mane_comparison.mp4').is_file():
        paths.append(root / 'mane_comparison.mp4')
    videos = {}
    for path in paths:
        metadata = json.loads(command(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)]))['streams'][0]
        assert metadata['avg_frame_rate'] == f"{report['fps']}/1" and int(metadata['nb_read_frames']) == report['frames']
        command(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'])
        videos[str(path.relative_to(root))] = metadata
    frames = sorted((root / 'optimized/frames').glob('*.png'))
    assert len(frames) == report['frames']
    unique = len({hashlib.sha256(path.read_bytes()).hexdigest() for path in frames})
    assert unique == len(frames), 'Repeated rendered frames'
    assert report['original_inputs_unchanged'] and report['learned_arrays_unchanged']
    assert report['max_root_error'] < 1e-6 and report['max_relative_length_error'] < 2e-4
    assert max(row['core_max_255'] for row in report['samples']) == 0
    if config['version'] == 'v3':
        coverage = config['coverage']
        assert coverage['graded_active_gaussians'] > coverage['accepted_mane_gaussians']
        assert coverage['new_dynamic_cleanup_candidates'] == 0
        if len(frames) > 1:
            assert all(row['temporal_std_mean'] > 0 for row in report['region_motion'].values())
    if args.against:
        for path in frames:
            with Image.open(path) as replay, Image.open(args.against / 'optimized/frames' / path.name) as original:
                assert replay.size == original.size
                assert ImageChops.difference(replay.convert('RGB'), original.convert('RGB')).getbbox() is None, path.name
    result = dict(version=config['version'], videos=videos, decoded_all_frames=True,
                  unique_rendered_frames=unique, archived_pixel_match_frames=len(frames) if args.against else 0,
                  max_root_error=report['max_root_error'], max_relative_length_error=report['max_relative_length_error'],
                  max_contact_residual=report['max_contact_residual'], regional_motion=report.get('region_motion'))
    value = json.dumps(result, indent=2) + '\n'
    (root / 'validation.json').write_text(value, encoding='utf-8')
    print(value, end='')


if __name__ == '__main__':
    main()
