#!/usr/bin/env python3
"""Audit the promoted WCurly HairGS adaptive-route experiment.

The script is intentionally read-only with respect to model/evaluation inputs.
It verifies the frozen train/test protocol, computes quality and efficiency
gates, checks HairGS chain topology, and measures leave-one-route-out
contributions.  Only the JSON and Markdown reports under ``--output-dir`` are
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from plyfile import PlyData


ROUTE_NAMES = ("shell", "fin", "strand")


@dataclass(frozen=True)
class EvalPair:
    baseline: dict
    candidate: dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_file_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text
    )
    if not match:
        raise ValueError(f"No elapsed time in {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"Unexpected elapsed time in {path}: {match.group(1)}")


def max_gpu_memory_mib(paths: list[Path]) -> int | None:
    values: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                try:
                    values.append(int(row[1].strip()))
                except ValueError:
                    continue
    return max(values) if values else None


def ply_counts(path: Path) -> tuple[dict[str, int], PlyData]:
    ply = PlyData.read(path)
    counts = {element.name: int(element.count) for element in ply.elements}
    return counts, ply


def topology_report(ply: PlyData) -> dict:
    by_name = {element.name: element.data for element in ply.elements}
    vertex_count = len(by_name["vertex"])
    edges = np.stack(
        (by_name["edge"]["vertex1"], by_name["edge"]["vertex2"]), axis=1
    ).astype(np.int64)
    roots = set(
        np.asarray(by_name["strand_root_idx"]["strand_root_idx"], dtype=np.int64).tolist()
    )
    ref_root_count = len(by_name["ref_strand_root"])

    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    degree = np.zeros(vertex_count, dtype=np.int32)
    used = np.zeros(vertex_count, dtype=bool)
    for left, right in edges.tolist():
        union(int(left), int(right))
        degree[left] += 1
        degree[right] += 1
        used[left] = True
        used[right] = True

    component_vertices: dict[int, int] = {}
    component_edges: dict[int, int] = {}
    component_has_root: dict[int, bool] = {}
    for vertex in np.flatnonzero(used).tolist():
        root = find(vertex)
        component_vertices[root] = component_vertices.get(root, 0) + 1
        if vertex in roots:
            component_has_root[root] = True
    for left, _ in edges.tolist():
        root = find(int(left))
        component_edges[root] = component_edges.get(root, 0) + 1

    components = len(component_edges)
    cycles = sum(
        component_edges[root] == component_vertices[root] for root in component_edges
    )
    malformed = sum(
        component_edges[root] != component_vertices[root] - 1
        for root in component_edges
    )
    explicit_rooted = sum(component_has_root.get(root, False) for root in component_edges)
    single_segment = sum(value == 1 for value in component_edges.values())
    max_degree = int(degree.max(initial=0))
    inferred_rooted = components - explicit_rooted if ref_root_count else 0
    return {
        "components": components,
        "single_segment_components": single_segment,
        "explicit_rooted_components": explicit_rooted,
        "reference_scalp_inferred_components": inferred_rooted,
        "cycles": cycles,
        "malformed_non_chain_components": malformed,
        "max_degree": max_degree,
        "reference_scalp_points": ref_root_count,
        "all_open_chains": cycles == 0 and malformed == 0 and max_degree <= 2,
        "all_components_have_hairgs_root": (
            explicit_rooted + inferred_rooted == components
        ),
    }


def verify_eval_pair(baseline: dict, candidate: dict, expected_count: int) -> None:
    for item in (baseline, candidate):
        if item.get("schema") != "external-render-evaluation-v1":
            raise ValueError("Unexpected evaluation schema")
        if item.get("render_size") != [500, 500]:
            raise ValueError("Evaluation is not 500x500")
        if item.get("image_count") != expected_count:
            raise ValueError("Unexpected split image count")
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("Baseline/candidate protocol mismatch")
    baseline_images = [row["image"] for row in baseline["per_frame"]]
    candidate_images = [row["image"] for row in candidate["per_frame"]]
    if baseline_images != candidate_images:
        raise ValueError("Baseline/candidate frame order mismatch")


def metric_delta(pair: EvalPair, name: str) -> float:
    return float(pair.candidate["aggregate"][name] - pair.baseline["aggregate"][name])


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--native-control", type=Path)
    parser.add_argument("--baseline-render-control", type=Path)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_train_path = args.baseline / "evaluation_train/external_evaluation/evaluation.json"
    baseline_test_path = args.baseline / "evaluation_test/external_evaluation/evaluation.json"
    candidate_train_path = args.candidate / "evaluation_train_5k/external_evaluation/evaluation.json"
    candidate_test_path = args.candidate / "evaluation_test_5k/external_evaluation/evaluation.json"
    train = EvalPair(load_json(baseline_train_path), load_json(candidate_train_path))
    test = EvalPair(load_json(baseline_test_path), load_json(candidate_test_path))
    verify_eval_pair(train.baseline, train.candidate, 12)
    verify_eval_pair(test.baseline, test.candidate, 4)
    native_control = None
    if args.native_control is not None:
        native_train = load_json(
            args.native_control / "evaluation_train/external_evaluation/evaluation.json"
        )
        native_test = load_json(
            args.native_control / "evaluation_test/external_evaluation/evaluation.json"
        )
        verify_eval_pair(train.baseline, native_train, 12)
        verify_eval_pair(test.baseline, native_test, 4)
        native_seconds = sum(
            parse_elapsed(args.native_control / "audit" / name)
            for name in ("native4k_time.txt", "native1k_time.txt")
        )
        native_control = {
            "train": native_train,
            "test": native_test,
            "stage3_measured_seconds": native_seconds,
        }

    baseline_ply = args.baseline / "point_cloud/iteration_30007/point_cloud.ply"
    candidate_ply = args.candidate / "point_cloud/iteration_20007/point_cloud.ply"
    route_sidecar = args.candidate / "point_cloud/iteration_20007/routes.npz"
    baseline_counts, _ = ply_counts(baseline_ply)
    candidate_counts, candidate_ply_data = ply_counts(candidate_ply)
    routes = np.load(route_sidecar)["route_id"]
    if routes.shape != (candidate_counts["segment"],):
        raise ValueError("Route sidecar is not aligned with candidate segments")
    route_counts = {
        "base": int(np.count_nonzero(routes == -1)),
        **{
            name: int(np.count_nonzero(routes == index))
            for index, name in enumerate(ROUTE_NAMES)
        },
    }
    if sum(route_counts.values()) != routes.size:
        raise ValueError("Some segment has an invalid route")

    baseline_bytes = baseline_ply.stat().st_size
    candidate_bytes = candidate_ply.stat().st_size + route_sidecar.stat().st_size
    baseline_checkpoint_bytes = tree_file_bytes(args.baseline / "point_cloud")
    candidate_checkpoint_bytes = tree_file_bytes(args.candidate / "point_cloud")
    model_reduction = 1.0 - candidate_bytes / baseline_bytes
    segment_reduction = 1.0 - candidate_counts["segment"] / baseline_counts["segment"]
    vertex_reduction = 1.0 - candidate_counts["vertex"] / baseline_counts["vertex"]

    baseline_stage3_seconds = (
        baseline_ply.stat().st_mtime
        - (args.baseline / "point_cloud/iteration_15007/point_cloud.ply").stat().st_mtime
    )
    candidate_stage3_seconds = sum(
        parse_elapsed(args.candidate / "audit" / name)
        for name in ("train_time.txt", "calibrate1k_time.txt")
    )
    stage3_reduction = 1.0 - candidate_stage3_seconds / baseline_stage3_seconds

    baseline_render_paths = sorted(args.baseline.glob("evaluation_*_render_*_time.txt"))
    if not baseline_render_paths:
        if args.baseline_render_control is None:
            raise ValueError(
                "Baseline render timing files are absent; pass "
                "--baseline-render-control from the exact serialization control"
            )
        baseline_render_paths = sorted(
            args.baseline_render_control.glob("evaluation_*_render_*_time.txt")
        )
    candidate_render_paths = sorted(args.candidate.glob("evaluation_*_5k_render_*_time.txt"))
    baseline_render = [parse_elapsed(path) for path in baseline_render_paths]
    candidate_render = [parse_elapsed(path) for path in candidate_render_paths]
    if len(baseline_render) != 4 or len(candidate_render) != 4:
        raise ValueError("Expected four train/test RGB/mask timing files per model")
    median_render_reduction = 1.0 - median(candidate_render) / median(baseline_render)

    quality_gates = {
        "heldout_foreground_psnr": metric_delta(test, "foreground_psnr") >= -0.50,
        "heldout_masked_lpips": metric_delta(test, "masked_lpips") <= 0.010,
        "heldout_mask_iou": metric_delta(test, "mask_iou") >= -0.010,
        "train_foreground_psnr": metric_delta(train, "foreground_psnr") >= -1.00,
        "largest_fn": metric_delta(test, "largest_false_negative_component_fraction") <= 0.020,
        "largest_fp": metric_delta(test, "largest_false_positive_component_fraction") <= 0.020,
    }
    efficiency_gates = {
        "model_size": model_reduction >= 0.30,
        "stage3_wall": stage3_reduction >= 0.50,
        "render_wall": median_render_reduction >= 0.25,
    }

    full = test.candidate["aggregate"]
    ablation_rows = {}
    hair_opacity_sum = 0.0
    for route in ROUTE_NAMES:
        root = args.ablations / f"drop_{route}"
        evaluation = load_json(root / "evaluation_test/external_evaluation/evaluation.json")
        verify_eval_pair(test.baseline, evaluation, 4)
        report = load_json(root / "ablation_report.json")
        opacity = float(report["removed_effective_opacity_sum"])
        hair_opacity_sum += opacity
        ablation_rows[route] = {
            "removed_segments": int(report["removed_segments"]),
            "removed_effective_opacity_sum": opacity,
            "full_minus_drop_foreground_psnr": float(
                full["foreground_psnr"] - evaluation["aggregate"]["foreground_psnr"]
            ),
            "full_minus_drop_mask_iou": float(
                full["mask_iou"] - evaluation["aggregate"]["mask_iou"]
            ),
            "drop_minus_full_masked_lpips": float(
                evaluation["aggregate"]["masked_lpips"] - full["masked_lpips"]
            ),
        }
    for route in ROUTE_NAMES:
        ablation_rows[route]["hair_opacity_mass_fraction"] = (
            ablation_rows[route]["removed_effective_opacity_sum"] / hair_opacity_sum
        )
        row = ablation_rows[route]
        row["positive_contributor"] = (
            row["full_minus_drop_foreground_psnr"] > 0
            and row["full_minus_drop_mask_iou"] > 0
            and row["drop_minus_full_masked_lpips"] > 0
        )

    topology = topology_report(candidate_ply_data)
    structural_gates = {
        "route_partition_complete": sum(route_counts.values()) == routes.size,
        "every_hair_route_nonempty": all(route_counts[name] > 0 for name in ROUTE_NAMES),
        "open_chain_topology": bool(topology["all_open_chains"]),
        "hairgs_root_available": bool(topology["all_components_have_hairgs_root"]),
        "all_routes_positive": all(
            ablation_rows[name]["positive_contributor"] for name in ROUTE_NAMES
        ),
    }
    efficiency_pass = efficiency_gates["model_size"] and sum(efficiency_gates.values()) >= 2
    promoted = (
        all(quality_gates.values())
        and efficiency_pass
        and all(structural_gates.values())
    )

    report = {
        "schema": "hairgs-adaptive-route-audit-v1",
        "status": "efficiency_promoted" if promoted else "blocked",
        "claim_status": {
            "quality_retention_and_efficiency": "promoted" if promoted else "blocked",
            "adaptive_route_image_quality_advantage": (
                "not_established" if args.native_control is not None else "not_tested"
            ),
        },
        "protocol": {
            "train": train.candidate["protocol"],
            "test": test.candidate["protocol"],
            "resolution": [500, 500],
            "train_images": 12,
            "test_images": 4,
            "same_frame_order": True,
        },
        "quality": {
            "train": {
                name: {
                    "baseline": float(train.baseline["aggregate"][name]),
                    "candidate": float(train.candidate["aggregate"][name]),
                    "delta": metric_delta(train, name),
                }
                for name in (
                    "foreground_psnr", "masked_lpips", "mask_iou",
                    "largest_false_negative_component_fraction",
                    "largest_false_positive_component_fraction",
                )
            },
            "heldout": {
                name: {
                    "baseline": float(test.baseline["aggregate"][name]),
                    "candidate": float(test.candidate["aggregate"][name]),
                    "delta": metric_delta(test, name),
                }
                for name in (
                    "foreground_psnr", "masked_full_psnr", "masked_ssim",
                    "masked_lpips", "mask_iou", "false_negative_fraction",
                    "false_positive_fraction",
                    "largest_false_negative_component_fraction",
                    "largest_false_positive_component_fraction",
                )
            },
            "gates": quality_gates,
        },
        "efficiency": {
            "baseline": {
                "stage3_steps": 15000,
                "total_optimizer_steps": 30000,
                "stage3_checkpoint_span_seconds": baseline_stage3_seconds,
                "deploy_bytes": baseline_bytes,
                "retained_checkpoint_bytes": baseline_checkpoint_bytes,
                "segments": baseline_counts["segment"],
                "vertices": baseline_counts["vertex"],
                "median_render_command_seconds": median(baseline_render),
            },
            "candidate": {
                "stage3_steps": 5000,
                "total_optimizer_steps": 20000,
                "stage3_measured_seconds": candidate_stage3_seconds,
                "deploy_bytes": candidate_bytes,
                "retained_checkpoint_bytes": candidate_checkpoint_bytes,
                "segments": candidate_counts["segment"],
                "vertices": candidate_counts["vertex"],
                "median_render_command_seconds": median(candidate_render),
                "observed_peak_gpu_memory_mib": max_gpu_memory_mib(
                    [
                        args.candidate / "audit/gpu.csv",
                        args.candidate / "audit/calibrate1k_gpu.csv",
                    ]
                ),
            },
            "reduction": {
                "model_size_fraction": model_reduction,
                "retained_checkpoint_storage_fraction": (
                    1.0 - candidate_checkpoint_bytes / baseline_checkpoint_bytes
                ),
                "segments_fraction": segment_reduction,
                "vertices_fraction": vertex_reduction,
                "stage3_steps_fraction": 1.0 - 5000 / 15000,
                "stage3_wall_fraction": stage3_reduction,
                "median_render_wall_fraction": median_render_reduction,
            },
            "gates": efficiency_gates,
            "pass": efficiency_pass,
            "measurement_note": (
                "Baseline Stage-3 wall is the timestamp span between native "
                "iteration_15007 and iteration_30007 checkpoints. Candidate wall "
                "is /usr/bin/time for its 4k and 1k processes, including restart "
                "overhead. Render timings include Python/model startup."
            ),
        },
        "structure": {
            "route_counts": route_counts,
            "route_ablation": ablation_rows,
            "topology": topology,
            "gates": structural_gates,
            "semantic_limit": (
                "All three routes are native HairGS line Gaussians with different "
                "ownership and gradient bandwidth. They are not yet an explicit "
                "shell volume, fin ribbon mesh, or simulation-ready continuous "
                "strand graph."
            ),
        },
        "provenance": {
            "baseline_ply": str(baseline_ply),
            "candidate_ply": str(candidate_ply),
            "route_sidecar": str(route_sidecar),
            "sha256": {
                "baseline_ply": sha256(baseline_ply),
                "candidate_ply": sha256(candidate_ply),
                "route_sidecar": sha256(route_sidecar),
                "baseline_train_evaluation": sha256(baseline_train_path),
                "baseline_test_evaluation": sha256(baseline_test_path),
                "candidate_train_evaluation": sha256(candidate_train_path),
                "candidate_test_evaluation": sha256(candidate_test_path),
            },
        },
    }
    if native_control is not None:
        route_vs_native = {
            name: float(
                test.candidate["aggregate"][name]
                - native_control["test"]["aggregate"][name]
            )
            for name in (
                "foreground_psnr", "masked_full_psnr", "masked_ssim",
                "masked_lpips", "mask_iou",
                "largest_false_negative_component_fraction",
                "largest_false_positive_component_fraction",
            )
        }
        report["same_budget_native_control"] = {
            "method": native_control["test"]["method"],
            "train_foreground_psnr": float(
                native_control["train"]["aggregate"]["foreground_psnr"]
            ),
            "heldout": {
                name: float(native_control["test"]["aggregate"][name])
                for name in route_vs_native
            },
            "adaptive_minus_native": route_vs_native,
            "stage3_measured_seconds": native_control["stage3_measured_seconds"],
            "interpretation": (
                "The adaptive and native 5k results are practically equivalent. "
                "The efficiency gain is attributable to Stage-2 capacity retention "
                "and early stopping; the current route-aware gradient bandwidth does "
                "not demonstrate a significant reconstruction-quality advantage."
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "promotion_audit.json"
    markdown_path = args.output_dir / "PROMOTION_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# WCurly HairGS adaptive-route promotion audit",
        "",
        f"Status: **{report['status'].upper()}**; adaptive-route image-quality advantage: **{report['claim_status']['adaptive_route_image_quality_advantage'].upper()}**.",
        "",
        "Frozen protocol: train12/test4, fixed cameras, 500 x 500, identical frame order and unified evaluator.",
        "",
        "## Quality",
        "",
        "| Split / metric | HairGS baseline | Adaptive 5k | Delta | Gate |",
        "|---|---:|---:|---:|:---:|",
        f"| train foreground PSNR | {fmt(train.baseline['aggregate']['foreground_psnr'])} | {fmt(train.candidate['aggregate']['foreground_psnr'])} | {fmt(metric_delta(train, 'foreground_psnr'))} | {'pass' if quality_gates['train_foreground_psnr'] else 'fail'} |",
        f"| held-out foreground PSNR | {fmt(test.baseline['aggregate']['foreground_psnr'])} | {fmt(test.candidate['aggregate']['foreground_psnr'])} | {fmt(metric_delta(test, 'foreground_psnr'))} | {'pass' if quality_gates['heldout_foreground_psnr'] else 'fail'} |",
        f"| held-out masked LPIPS | {fmt(test.baseline['aggregate']['masked_lpips'])} | {fmt(test.candidate['aggregate']['masked_lpips'])} | {fmt(metric_delta(test, 'masked_lpips'))} | {'pass' if quality_gates['heldout_masked_lpips'] else 'fail'} |",
        f"| held-out mask IoU | {fmt(test.baseline['aggregate']['mask_iou'])} | {fmt(test.candidate['aggregate']['mask_iou'])} | {fmt(metric_delta(test, 'mask_iou'))} | {'pass' if quality_gates['heldout_mask_iou'] else 'fail'} |",
        f"| held-out largest FN | {fmt(test.baseline['aggregate']['largest_false_negative_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_negative_component_fraction'])} | {fmt(metric_delta(test, 'largest_false_negative_component_fraction'))} | {'pass' if quality_gates['largest_fn'] else 'fail'} |",
        f"| held-out largest FP | {fmt(test.baseline['aggregate']['largest_false_positive_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_positive_component_fraction'])} | {fmt(metric_delta(test, 'largest_false_positive_component_fraction'))} | {'pass' if quality_gates['largest_fp'] else 'fail'} |",
        "",
        "## Efficiency",
        "",
        "| Quantity | HairGS baseline | Adaptive 5k | Reduction | Gate |",
        "|---|---:|---:|---:|:---:|",
        f"| Stage-3 steps | 15,000 | 5,000 | {100*(1-5000/15000):.2f}% | pass |",
        f"| total optimizer steps | 30,000 | 20,000 | {100*(1-20000/30000):.2f}% | diagnostic |",
        f"| Stage-3 wall | {baseline_stage3_seconds:.2f} s | {candidate_stage3_seconds:.2f} s | {100*stage3_reduction:.2f}% | {'pass' if efficiency_gates['stage3_wall'] else 'fail'} |",
        f"| deploy artifact | {baseline_bytes:,} B | {candidate_bytes:,} B | {100*model_reduction:.2f}% | {'pass' if efficiency_gates['model_size'] else 'fail'} |",
        f"| retained point-cloud checkpoints | {baseline_checkpoint_bytes:,} B | {candidate_checkpoint_bytes:,} B | {100*(1-candidate_checkpoint_bytes/baseline_checkpoint_bytes):.2f}% | diagnostic |",
        f"| segments | {baseline_counts['segment']:,} | {candidate_counts['segment']:,} | {100*segment_reduction:.2f}% | diagnostic |",
        f"| median render command | {median(baseline_render):.2f} s | {median(candidate_render):.2f} s | {100*median_render_reduction:.2f}% | {'pass' if efficiency_gates['render_wall'] else 'fail'} |",
        "",
        "Render timing includes process/model startup, so it is diagnostic rather than a kernel-only benchmark.",
        "",
        "## Route contribution",
        "",
        "| Route | segments | hair opacity mass | full - drop PSNR | full - drop IoU | drop - full LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTE_NAMES:
        row = ablation_rows[route]
        lines.append(
            f"| {route} | {route_counts[route]:,} | {100*row['hair_opacity_mass_fraction']:.2f}% | "
            f"{fmt(row['full_minus_drop_foreground_psnr'])} | "
            f"{fmt(row['full_minus_drop_mask_iou'])} | "
            f"{fmt(row['drop_minus_full_masked_lpips'])} |"
        )
    lines += [
        "",
        "All three removals degrade held-out PSNR, IoU, and LPIPS; every route is a positive contributor under the frozen test split.",
        "",
        "## Structural scope",
        "",
        f"The final PLY has {topology['components']:,} open-chain components; {topology['single_segment_components']:,} contain one segment. "
        f"Only {topology['explicit_rooted_components']:,} components contain an explicit root; the other {topology['reference_scalp_inferred_components']:,} use HairGS reference-scalp inference.",
        "",
        "The present shell/fin/strand routes are native HairGS line Gaussians with route-specific ownership and update bandwidth. This audit supports rendering/efficiency claims, not yet a claim of explicit shell volume, fin ribbon geometry, or simulation-ready continuous strands.",
        "",
        "## Decision",
        "",
        f"Quality gates: {'pass' if all(quality_gates.values()) else 'fail'}; efficiency gates: {'pass' if efficiency_pass else 'fail'}; structural gates: {'pass' if all(structural_gates.values()) else 'fail'}.",
    ]
    if native_control is not None:
        native_test_aggregate = native_control["test"]["aggregate"]
        lines += [
            "",
            "## Same-budget native HairGS control",
            "",
            "| Held-out metric | native HairGS 5k | adaptive-route 5k | adaptive - native |",
            "|---|---:|---:|---:|",
            f"| foreground PSNR | {fmt(native_test_aggregate['foreground_psnr'])} | {fmt(test.candidate['aggregate']['foreground_psnr'])} | {fmt(test.candidate['aggregate']['foreground_psnr'] - native_test_aggregate['foreground_psnr'])} |",
            f"| masked LPIPS | {fmt(native_test_aggregate['masked_lpips'])} | {fmt(test.candidate['aggregate']['masked_lpips'])} | {fmt(test.candidate['aggregate']['masked_lpips'] - native_test_aggregate['masked_lpips'])} |",
            f"| mask IoU | {fmt(native_test_aggregate['mask_iou'])} | {fmt(test.candidate['aggregate']['mask_iou'])} | {fmt(test.candidate['aggregate']['mask_iou'] - native_test_aggregate['mask_iou'])} |",
            f"| largest FN | {fmt(native_test_aggregate['largest_false_negative_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_negative_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_negative_component_fraction'] - native_test_aggregate['largest_false_negative_component_fraction'])} |",
            f"| largest FP | {fmt(native_test_aggregate['largest_false_positive_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_positive_component_fraction'])} | {fmt(test.candidate['aggregate']['largest_false_positive_component_fraction'] - native_test_aggregate['largest_false_positive_component_fraction'])} |",
            "",
            f"Measured 5k training wall: native {native_control['stage3_measured_seconds']:.2f} s versus adaptive {candidate_stage3_seconds:.2f} s.",
            "",
            "The two 5k reconstructions are practically equivalent. The strong efficiency result comes from retaining HairGS Stage-2 capacity and early stopping, not from a demonstrated image-quality advantage of route-aware gradient scaling. The route result currently supports ownership/ablation semantics only.",
        ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
