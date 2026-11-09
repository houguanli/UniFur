"""Replay fingerprints, pose access, silhouette diagnostics and video encoding."""
import hashlib
import shutil
import subprocess

import numpy as np

def fingerprint(paths):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def pose_of(renderer):
    if renderer._collision_pose is not None:
        return renderer._collision_pose
    if renderer._world_at_rest:
        return dict(xyz=renderer.rest_xyz, rotation=renderer.rest_rotation, nodes=renderer.rest_nodes)
    return dict(xyz=renderer._world_xyz, rotation=renderer._world_rotation, nodes=renderer._world_nodes)


def measurements(a,b,guard):
    x,y=np.asarray(a,float)/255,np.asarray(b,float)/255
    diff=np.abs(x-y)
    luma=np.asarray([.2126,.7152,.0722])
    bg=np.asarray([.94,.95,.97])@luma
    outside=guard['weight']>.999
    # This is a fixed A/B proxy region from the unmodified trusted subset,
    # not a claim of ground-truth segmentation or real-fur quality.
    da=np.maximum(0,bg-x@luma); db=np.maximum(0,bg-y@luma)
    return dict(mae_255=float(diff.mean()*255),core_max_255=float(diff[guard['core']].max()*255),
        exterior_dark_energy_before=float(da[outside].sum()),exterior_dark_energy_after=float(db[outside].sum()),
        exterior_dark_pixels_before=int((da[outside]>.04).sum()),exterior_dark_pixels_after=int((db[outside]>.04).sum()),
        exterior_pixel_count=int(outside.sum()))

def make_video(output, fps, preview):
    executable = shutil.which("ffmpeg")
    if not executable:
        return {"status": "ffmpeg_unavailable", "frames_retained": True}
    video = output / "preview.mp4"
    command = [executable, "-hide_banner", "-loglevel", "error", "-n", "-framerate", str(fps),
               "-i", str(output/"frames"/"%05d.png"), "-c:v", "libx264", "-crf", "19",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    if preview:
        # A compact browser-readable silent loop; preserve full resolution in MP4.
        subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-n", "-i", str(video),
                        "-filter_complex", "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
                        "-loop", "0", str(output/"preview.gif")],
                       check=True, capture_output=True, timeout=120)
    return {"status": "complete", "video": str(video), "gif": preview}
