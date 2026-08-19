from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .fiber_evaluate import evaluate_unified_fiber_stage2
from .fiber_optimize import optimize_unified_fiber_stage2
from .fiber_route_audit import audit_unified_fiber_routes


def _add_protocol_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1-npz", required=True, help="Surface/motion scaffold NPZ.")
    parser.add_argument("--gaussian-ply", required=True, help="Initial Gaussian scaffold PLY.")
    parser.add_argument(
        "--fixed-base-gaussian-ply",
        default=None,
        help="Optional immutable head/body Gaussian base.",
    )
    parser.add_argument("--frame-dir", required=True, help="RGBA/RGB target images.")
    parser.add_argument(
        "--camera-manifest",
        default=None,
        help="Per-view camera and optional motion manifest.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--renderer", choices=["torch", "hairgs"], default=None)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unifur",
        description="Unified shell/strand Gaussian reconstruction for fur and hair.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("fiber-stage2", help="Optimize a unified fiber field.")
    _add_protocol_inputs(train)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--max-points", type=int, default=None)
    train.add_argument("--log-every", type=int, default=None)
    train.add_argument("--checkpoint-every", type=int, default=None)
    train.add_argument(
        "--residual-bootstrap-checkpoint",
        default=None,
        help="Optional photometric 3DGS teacher used only for initialization.",
    )
    train.add_argument("--fixed-base-max-scale-fraction", type=float, default=None)

    evaluate = sub.add_parser("fiber-eval", help="Render and evaluate a trained field.")
    _add_protocol_inputs(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--route-mode",
        choices=["hard", "soft", "shell", "strand", "residual"],
        default="hard",
    )
    evaluate.add_argument("--residual-max-scale-fraction", type=float, default=None)
    evaluate.add_argument("--fixed-base-max-scale-fraction", type=float, default=None)
    evaluate.add_argument("--export-external-renders", action="store_true")

    audit = sub.add_parser(
        "fiber-route-audit",
        help="Audit route confidence and leave-one-route-out contribution.",
    )
    _add_protocol_inputs(audit)
    audit.add_argument("--checkpoint", required=True)
    return parser


def _render_size(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.render_width is None and args.render_height is None:
        return None
    if args.render_width is None or args.render_height is None:
        raise ValueError("--render-width and --render-height must be supplied together")
    return int(args.render_width), int(args.render_height)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(config_path if config_path.is_file() else None)

    if args.command == "fiber-stage2":
        if args.fixed_base_max_scale_fraction is not None:
            cfg.fiber_fixed_base_max_scale_fraction = float(
                args.fixed_base_max_scale_fraction
            )
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
            render_size=_render_size(args),
            camera_manifest=args.camera_manifest,
            residual_bootstrap_checkpoint=args.residual_bootstrap_checkpoint,
            fixed_base_gaussian_ply=args.fixed_base_gaussian_ply,
        )
        print(f"checkpoint={artifacts.checkpoint_pt}")
        print(f"state={artifacts.state_npz}")
        print(f"report={artifacts.report_json}")
        print(f"metrics={artifacts.metrics_jsonl}")
        print(f"loss_curve={artifacts.loss_curve_png}")
        return 0

    if args.command == "fiber-eval":
        if args.residual_max_scale_fraction is not None:
            cfg.fiber_residual_max_scale_fraction = float(
                args.residual_max_scale_fraction
            )
        if args.fixed_base_max_scale_fraction is not None:
            cfg.fiber_fixed_base_max_scale_fraction = float(
                args.fixed_base_max_scale_fraction
            )
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
            render_size=_render_size(args),
            camera_manifest=args.camera_manifest,
            export_external_renders=args.export_external_renders,
            fixed_base_gaussian_ply=args.fixed_base_gaussian_ply,
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
            render_size=_render_size(args),
            camera_manifest=args.camera_manifest,
            fixed_base_gaussian_ply=args.fixed_base_gaussian_ply,
        )
        print(f"route_audit={artifacts.report_json}")
        print(f"route_audit_plot={artifacts.plot_png}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
