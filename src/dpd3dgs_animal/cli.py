from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .coordinate_calibration import rank_axis_transforms
from .diagnostics import audit_pipeline
from .elastic_bridge import run_elastic_forward
from .fem_optimize import optimize_constrained_fem_stage2
from .fiber_evaluate import evaluate_unified_fiber_stage2
from .fiber_optimize import optimize_unified_fiber_stage2
from .fiber_route_audit import audit_unified_fiber_routes
from .mocap_adapter import MocapAnythingAdapter
from .optimize import optimize_stage2
from .pipeline import run_stage1
from .preprocess import video_to_transparent_frames
from .sam3d_adapter import mesh_to_arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpd3dgs-animal")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    stage1 = sub.add_parser("stage1", help="Run video -> SAM3D/Mocap -> tet/surface/GS/loss stage.")
    stage1.add_argument("--video", required=True, help="Input monocular animal motion video or frame folder.")
    stage1.add_argument("--work-dir", required=True, help="Output work directory.")
    stage1.add_argument("--mask", default=None, help="Optional reference-frame object mask for SAM3D.")
    stage1.add_argument(
        "--sam3d-reference-image",
        default=None,
        help="SAM3D RGB crop corresponding exactly to --mask.",
    )
    stage1.add_argument(
        "--sam3d-reference-transform",
        default=None,
        help="JSON mapping the SAM3D crop back to full-frame pixels.",
    )
    stage1.add_argument(
        "--sam3d-metadata",
        default=None,
        help="SAM3D camera metadata for existing camera-space mesh/PLY assets.",
    )
    stage1.add_argument("--mesh", default=None, help="Optional existing mesh path to skip SAM3D mesh extraction.")
    stage1.add_argument("--gaussian-ply", default=None, help="Optional existing 3DGS PLY path.")
    stage1.add_argument("--mocap-prediction", default=None, help="Optional existing MocapAnything *_pred.npy.")
    stage1.add_argument("--skip-sam3d", action="store_true")
    stage1.add_argument("--skip-mocap", action="store_true")
    stage1.add_argument("--max-frames", type=int, default=None)
    stage1.add_argument("--render-width", type=int, default=None, help="Override render width. Defaults to input frame width.")
    stage1.add_argument("--render-height", type=int, default=None, help="Override render height. Defaults to input frame height.")

    stage2 = sub.add_parser("stage2", help="Optimize per-frame skeleton nodes through differentiable render loss.")
    stage2.add_argument("--stage1-npz", required=True, help="Stage 1 tet/skeleton/surface NPZ.")
    stage2.add_argument("--gaussian-ply", required=True, help="SAM3D Gaussian PLY.")
    stage2.add_argument("--frame-dir", required=True, help="Ground-truth frame directory.")
    stage2.add_argument("--out-dir", required=True)
    stage2.add_argument("--steps", type=int, default=None)
    stage2.add_argument("--lr", type=float, default=None)
    stage2.add_argument("--max-frames", type=int, default=None)
    stage2.add_argument("--render-width", type=int, default=None, help="Override render width. Defaults to frame/stage1 width.")
    stage2.add_argument("--render-height", type=int, default=None, help="Override render height. Defaults to frame/stage1 height.")

    elastic = sub.add_parser("elastic-forward", help="Run ElasticSimulator constrained FEM forward pass from Stage 1 artifacts.")
    elastic.add_argument("--stage1-npz", required=True, help="Stage 1 tet/skeleton/surface NPZ.")
    elastic.add_argument("--out-dir", required=True)
    elastic.add_argument("--gaussian-ply", default=None, help="Optional SAM3D Gaussian PLY for render/loss evaluation.")
    elastic.add_argument("--frame-dir", default=None, help="Optional ground-truth frame directory for render/loss evaluation.")
    elastic.add_argument("--driver", default=None, help="Optional HeadlessConstrainedFEM binary path.")
    elastic.add_argument("--max-frames", type=int, default=None)
    elastic.add_argument("--radius-scale", type=float, default=None)
    elastic.add_argument("--min-radius-scale", type=float, default=None)
    elastic.add_argument("--max-radius-scale", type=float, default=None)
    elastic.add_argument("--min-bone-length-scale", type=float, default=None)
    elastic.add_argument("--render-width", type=int, default=None, help="Override render width. Defaults to frame/stage1 width.")
    elastic.add_argument("--render-height", type=int, default=None, help="Override render height. Defaults to frame/stage1 height.")

    elastic_stage2 = sub.add_parser(
        "elastic-stage2",
        help="Optimize skeleton nodes through differentiable constrained tetrahedral FEM and render loss.",
    )
    elastic_stage2.add_argument("--stage1-npz", required=True, help="Stage 1 tet/skeleton/surface NPZ.")
    elastic_stage2.add_argument("--gaussian-ply", required=True, help="SAM3D Gaussian PLY.")
    elastic_stage2.add_argument("--frame-dir", required=True, help="Ground-truth frame directory.")
    elastic_stage2.add_argument("--out-dir", required=True)
    elastic_stage2.add_argument("--steps", type=int, default=None)
    elastic_stage2.add_argument("--lr", type=float, default=None)
    elastic_stage2.add_argument("--max-frames", type=int, default=None)
    elastic_stage2.add_argument("--render-width", type=int, default=None, help="Override render width. Defaults to frame/stage1 width.")
    elastic_stage2.add_argument("--render-height", type=int, default=None, help="Override render height. Defaults to frame/stage1 height.")

    fiber_stage2 = sub.add_parser(
        "fiber-stage2",
        help="Optimize the surface-anchored shell/strand/residual Gaussian field.",
    )
    fiber_stage2.add_argument("--stage1-npz", required=True, help="Stage 1 tet/skeleton/surface NPZ.")
    fiber_stage2.add_argument("--gaussian-ply", required=True, help="Initial ordinary 3DGS/SAM3D Gaussian PLY.")
    fiber_stage2.add_argument("--frame-dir", required=True, help="Ground-truth RGBA frame directory.")
    fiber_stage2.add_argument(
        "--camera-manifest",
        default=None,
        help=(
            "Optional per-image camera/motion JSON. Enables static or dynamic "
            "multi-view training; omitted keeps the legacy fixed-camera sequence."
        ),
    )
    fiber_stage2.add_argument("--out-dir", required=True)
    fiber_stage2.add_argument("--steps", type=int, default=None)
    fiber_stage2.add_argument("--lr", type=float, default=None)
    fiber_stage2.add_argument("--max-points", type=int, default=None)
    fiber_stage2.add_argument("--max-frames", type=int, default=None)
    fiber_stage2.add_argument("--frame-start", type=int, default=0)
    fiber_stage2.add_argument("--frame-stride", type=int, default=1)
    fiber_stage2.add_argument("--log-every", type=int, default=None)
    fiber_stage2.add_argument("--checkpoint-every", type=int, default=None)
    fiber_stage2.add_argument(
        "--renderer", choices=["torch", "hairgs"], default=None,
        help="Differentiable renderer. HairGS requires the hair-gs Conda environment.",
    )
    fiber_stage2.add_argument("--render-width", type=int, default=None)
    fiber_stage2.add_argument("--render-height", type=int, default=None)

    fiber_eval = sub.add_parser(
        "fiber-eval",
        help="Evaluate a trained unified fiber field on a held-out frame slice.",
    )
    fiber_eval.add_argument("--stage1-npz", required=True)
    fiber_eval.add_argument("--gaussian-ply", required=True)
    fiber_eval.add_argument("--checkpoint", required=True)
    fiber_eval.add_argument("--frame-dir", required=True)
    fiber_eval.add_argument(
        "--camera-manifest",
        default=None,
        help="Optional per-image camera/motion JSON used during evaluation.",
    )
    fiber_eval.add_argument("--out-dir", required=True)
    fiber_eval.add_argument("--max-frames", type=int, default=None)
    fiber_eval.add_argument("--frame-start", type=int, default=0)
    fiber_eval.add_argument("--frame-stride", type=int, default=1)
    fiber_eval.add_argument("--renderer", choices=["torch", "hairgs"], default=None)
    fiber_eval.add_argument(
        "--route-mode",
        choices=["hard", "soft", "shell", "strand", "residual"],
        default="hard",
        help="Evaluate the deployed hard route, training-time soft mixture, or one forced route.",
    )
    fiber_eval.add_argument("--render-width", type=int, default=None)
    fiber_eval.add_argument("--render-height", type=int, default=None)
    fiber_eval.add_argument(
        "--export-external-renders",
        action="store_true",
        help="Save per-frame RGB/alpha arrays and an external-evaluator manifest.",
    )

    fiber_audit = sub.add_parser(
        "fiber-route-audit",
        help="Audit route confidence, spatial coherence, and leave-one-route-out contribution.",
    )
    fiber_audit.add_argument("--stage1-npz", required=True)
    fiber_audit.add_argument("--gaussian-ply", required=True)
    fiber_audit.add_argument("--checkpoint", required=True)
    fiber_audit.add_argument("--frame-dir", required=True)
    fiber_audit.add_argument(
        "--camera-manifest",
        default=None,
        help="Optional per-image camera/motion JSON used by the route audit.",
    )
    fiber_audit.add_argument("--out-dir", required=True)
    fiber_audit.add_argument("--max-frames", type=int, default=None)
    fiber_audit.add_argument("--frame-start", type=int, default=0)
    fiber_audit.add_argument("--frame-stride", type=int, default=1)
    fiber_audit.add_argument("--renderer", choices=["torch", "hairgs"], default=None)
    fiber_audit.add_argument("--render-width", type=int, default=None)
    fiber_audit.add_argument("--render-height", type=int, default=None)

    prepare = sub.add_parser("prepare-video", help="Extract MP4 frames and infer alpha masks with local RMBG.")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--method", choices=["rmbg", "connected_background"], default="rmbg")
    prepare.add_argument("--background-threshold", type=float, default=10.0)

    calib = sub.add_parser("calibrate-coordinates", help="Rank Mocap->mesh axis transforms.")
    calib.add_argument("--mesh", required=True, help="SAM3D mesh path readable by trimesh.")
    calib.add_argument("--mocap-prediction", required=True, help="MocapAnything *_pred.npy.")
    calib.add_argument("--ref-seq", default=None, help="Reference sequence, e.g. Dog#Dog-Galloping/y30.")
    calib.add_argument("--out-json", default=None)

    diagnose = sub.add_parser("diagnose", help="Render stage-by-stage pipeline diagnostics.")
    diagnose.add_argument("--frame-dir", required=True)
    diagnose.add_argument("--gaussian-ply", required=True)
    diagnose.add_argument("--stage1-npz", required=True)
    diagnose.add_argument("--out-dir", required=True)
    diagnose.add_argument("--baseline-stage1-npz", default=None)
    diagnose.add_argument("--optimized-joints", default=None)
    diagnose.add_argument(
        "--frames",
        default="0,30,60,89",
        help="Comma-separated frame indices used in the diagnostic contact sheets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.command == "stage1":
        artifacts = run_stage1(
            input_video=args.video,
            work_dir=args.work_dir,
            cfg=cfg,
            mask_path=args.mask,
            mesh_path=args.mesh,
            gaussian_ply=args.gaussian_ply,
            mocap_prediction=args.mocap_prediction,
            run_sam3d=not args.skip_sam3d,
            run_mocap=not args.skip_mocap,
            max_frames=args.max_frames,
            render_size=_render_size_from_args(args),
            sam3d_reference_image=args.sam3d_reference_image,
            sam3d_reference_transform=args.sam3d_reference_transform,
            sam3d_metadata=args.sam3d_metadata,
        )
        print(f"summary={artifacts.summary_json}")
        print(f"tet_npz={artifacts.tet_npz}")
        if artifacts.losses_json:
            print(f"losses={artifacts.losses_json}")
        return 0
    if args.command == "stage2":
        artifacts = optimize_stage2(
            stage1_npz=args.stage1_npz,
            gaussian_ply=args.gaussian_ply,
            frame_dir=args.frame_dir,
            out_dir=args.out_dir,
            cfg=cfg,
            steps=args.steps,
            lr=args.lr,
            render_size=_render_size_from_args(args),
            max_frames=args.max_frames,
        )
        print(f"optimized_npz={artifacts.optimized_npz}")
        print(f"optimized_joints={artifacts.joints_npy}")
        print(f"losses={artifacts.losses_json}")
        return 0
    if args.command == "elastic-forward":
        artifacts = run_elastic_forward(
            stage1_npz=args.stage1_npz,
            out_dir=args.out_dir,
            cfg=cfg,
            gaussian_ply=args.gaussian_ply,
            frame_dir=args.frame_dir,
            max_frames=args.max_frames,
            driver_path=args.driver,
            render_size=_render_size_from_args(args),
            radius_scale=args.radius_scale,
            min_radius_scale=args.min_radius_scale,
            max_radius_scale=args.max_radius_scale,
            min_bone_length_scale=args.min_bone_length_scale,
        )
        print(f"state_npz={artifacts.state_npz}")
        print(f"vertices_bin={artifacts.vertices_bin}")
        print(f"driver_log={artifacts.driver_log}")
        if artifacts.losses_json:
            print(f"losses={artifacts.losses_json}")
        return 0
    if args.command == "elastic-stage2":
        artifacts = optimize_constrained_fem_stage2(
            stage1_npz=args.stage1_npz,
            gaussian_ply=args.gaussian_ply,
            frame_dir=args.frame_dir,
            out_dir=args.out_dir,
            cfg=cfg,
            steps=args.steps,
            lr=args.lr,
            render_size=_render_size_from_args(args),
            max_frames=args.max_frames,
        )
        print(f"optimized_npz={artifacts.optimized_npz}")
        print(f"optimized_joints={artifacts.joints_npy}")
        print(f"losses={artifacts.losses_json}")
        return 0
    if args.command == "fiber-stage2":
        artifacts = optimize_unified_fiber_stage2(
            stage1_npz=args.stage1_npz,
            gaussian_ply=args.gaussian_ply,
            frame_dir=args.frame_dir,
            out_dir=args.out_dir,
            cfg=cfg,
            steps=args.steps,
            lr=args.lr,
            max_points=args.max_points,
            renderer=args.renderer,
            max_frames=args.max_frames,
            frame_start=args.frame_start,
            frame_stride=args.frame_stride,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            render_size=_render_size_from_args(args),
            camera_manifest=args.camera_manifest,
        )
        print(f"checkpoint={artifacts.checkpoint_pt}")
        print(f"state={artifacts.state_npz}")
        print(f"report={artifacts.report_json}")
        print(f"metrics={artifacts.metrics_jsonl}")
        print(f"loss_curve={artifacts.loss_curve_png}")
        return 0
    if args.command == "fiber-eval":
        artifacts = evaluate_unified_fiber_stage2(
            stage1_npz=args.stage1_npz,
            gaussian_ply=args.gaussian_ply,
            checkpoint_pt=args.checkpoint,
            frame_dir=args.frame_dir,
            out_dir=args.out_dir,
            cfg=cfg,
            renderer=args.renderer,
            max_frames=args.max_frames,
            frame_start=args.frame_start,
            frame_stride=args.frame_stride,
            route_mode=args.route_mode,
            render_size=_render_size_from_args(args),
            camera_manifest=args.camera_manifest,
            export_external_renders=args.export_external_renders,
        )
        print(f"evaluation={artifacts.report_json}")
        print(f"contact_sheet={artifacts.contact_sheet_png}")
        return 0
    if args.command == "fiber-route-audit":
        artifacts = audit_unified_fiber_routes(
            stage1_npz=args.stage1_npz,
            gaussian_ply=args.gaussian_ply,
            checkpoint_pt=args.checkpoint,
            frame_dir=args.frame_dir,
            out_dir=args.out_dir,
            cfg=cfg,
            renderer=args.renderer,
            max_frames=args.max_frames,
            frame_start=args.frame_start,
            frame_stride=args.frame_stride,
            render_size=_render_size_from_args(args),
            camera_manifest=args.camera_manifest,
        )
        print(f"route_audit={artifacts.report_json}")
        print(f"route_audit_plot={artifacts.plot_png}")
        return 0
    if args.command == "prepare-video":
        artifacts = video_to_transparent_frames(
            args.video,
            args.out_dir,
            method=args.method,
            rmbg_weights_dir=cfg.paths.mocap_checkpoints / "RMBG-1.4",
            device=cfg.device,
            background_threshold=args.background_threshold,
        )
        print(f"frames={artifacts.frame_count}")
        print(f"size={artifacts.size[0]}x{artifacts.size[1]}")
        print(f"rgba_dir={artifacts.rgba_dir}")
        print(f"mask_dir={artifacts.mask_dir}")
        print(f"reference_mask={artifacts.ref_mask}")
        print(f"reference_transform={artifacts.ref_transform}")
        return 0
    if args.command == "calibrate-coordinates":
        vertices, _faces = mesh_to_arrays(args.mesh)
        mocap = MocapAnythingAdapter(
            cfg.paths.mocap_root,
            checkpoint_root=cfg.paths.mocap_checkpoints,
            zoo_root=cfg.paths.mocap_zoo,
        )
        ref_seq = args.ref_seq or cfg.mocap_ref_seq
        skeleton = mocap.load_skeleton_prior(args.mocap_prediction, ref_seq)
        scores = rank_axis_transforms(skeleton.joints, vertices)
        payload = {
            "best": scores[0].name,
            "scores": [score.__dict__ for score in scores],
        }
        if args.out_json:
            out = Path(args.out_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "diagnose":
        report = audit_pipeline(
            frame_dir=args.frame_dir,
            gaussian_ply=args.gaussian_ply,
            stage1_npz=args.stage1_npz,
            out_dir=args.out_dir,
            cfg=cfg,
            baseline_stage1_npz=args.baseline_stage1_npz,
            optimized_joints=args.optimized_joints,
            frame_indices=tuple(
                int(value.strip())
                for value in args.frames.split(",")
                if value.strip()
            ),
        )
        print(f"report={report}")
        return 0
    raise AssertionError(args.command)


def _render_size_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.render_width is None and args.render_height is None:
        return None
    if args.render_width is None or args.render_height is None:
        raise ValueError("--render-width and --render-height must be supplied together")
    return int(args.render_width), int(args.render_height)


if __name__ == "__main__":
    raise SystemExit(main())
