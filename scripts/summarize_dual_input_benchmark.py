#!/usr/bin/env python3
"""Build protocol-separated JSON/CSV/Markdown tables from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STATIC_RUNS = {
    "residual_only": (
        "eval_residual_v28_20k_r512_softmask/evaluation.json",
        "full_residual_v28_20k_r512_softmask/unified_fiber_report.json",
    ),
    "unified_soft": (
        "eval_unified_v28_20k_r512_softmask/evaluation.json",
        "full_unified_v28_20k_r512_softmask/unified_fiber_report.json",
    ),
}
DYNAMIC_METHODS = ("residual", "unified")
DYNAMIC_SETTINGS = ("mono", "mv4", "mv8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=Path("/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared"),
    )
    parser.add_argument(
        "--dynamic-root",
        type=Path,
        default=Path("/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual_results"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def load_evaluation(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_row(
    protocol: str,
    setting: str,
    method: str,
    payload: dict,
    comparable_group: str,
    training_report: dict | None = None,
) -> dict:
    aggregate = payload["aggregate"]
    row = {
        "protocol": protocol,
        "setting": setting,
        "method": method,
        "comparable_group": comparable_group,
        "status": "complete",
        "foreground_psnr": aggregate.get("foreground_psnr"),
        "masked_full_psnr": aggregate.get("masked_full_psnr"),
        "full_psnr": aggregate.get("full_psnr"),
        "full_ssim": aggregate.get("full_ssim"),
        "full_lpips": aggregate.get("full_lpips"),
        "mask_iou": aggregate.get("mask_iou"),
        "mask_f1": aggregate.get("mask_f1"),
        "background_opacity_mean": aggregate.get("background_opacity_mean"),
        "checkpoint": str(payload.get("checkpoint", "")),
    }
    routes = payload.get("soft_routes") or {}
    for route in ("shell", "strand", "residual"):
        row[f"route_{route}"] = routes.get(route)
    if training_report is not None:
        resources = training_report.get("resource_usage") or {}
        row.update(
            {
                "source_roots": training_report.get("points"),
                "train_seconds": resources.get("elapsed_training_seconds"),
                "peak_cuda_allocated_gb": (
                    resources.get("peak_cuda_allocated_bytes", 0) / (1024 ** 3)
                    if resources.get("peak_cuda_allocated_bytes") is not None
                    else None
                ),
                "peak_cuda_reserved_gb": (
                    resources.get("peak_cuda_reserved_bytes", 0) / (1024 ** 3)
                    if resources.get("peak_cuda_reserved_bytes") is not None
                    else None
                ),
            }
        )
    return row


def pending_row(protocol: str, setting: str, method: str, comparable_group: str, state: str) -> dict:
    return {
        "protocol": protocol,
        "setting": setting,
        "method": method,
        "comparable_group": comparable_group,
        "status": state,
    }


def collect(static_root: Path, dynamic_root: Path) -> list[dict]:
    rows: list[dict] = []
    for method, (eval_relative, report_relative) in STATIC_RUNS.items():
        path = static_root / eval_relative
        payload = load_evaluation(path)
        training_report = load_evaluation(static_root / report_relative)
        rows.append(
            metric_row(
                "S-mv-official-prior",
                "28-fit/8-test",
                method,
                payload,
                "static-released-all-view-body-prior-20k",
                training_report,
            )
            if payload is not None
            else pending_row(
                "S-mv-official-prior",
                "28-fit/8-test",
                method,
                "static-released-all-view-body-prior-20k",
                "missing",
            )
        )

    for setting in DYNAMIC_SETTINGS:
        for method in DYNAMIC_METHODS:
            path = dynamic_root / f"{setting}_{method}_20k_eval_novel_v8/evaluation.json"
            payload = load_evaluation(path)
            training_report = load_evaluation(
                dynamic_root / f"{setting}_{method}_20k/unified_fiber_report.json"
            )
            group = f"DFA-Panda-Walk-32f-v1-{setting}-neutral-template-20k"
            rows.append(
                metric_row(
                    "DFA-Panda-Walk-32f-v1",
                    setting,
                    "residual_only" if method == "residual" else "unified_soft",
                    payload,
                    group,
                    training_report,
                )
                if payload is not None
                else pending_row(
                    "DFA-Panda-Walk-32f-v1",
                    setting,
                    "residual_only" if method == "residual" else "unified_soft",
                    group,
                    "queued",
                )
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "protocol", "setting", "method", "comparable_group", "status",
        "foreground_psnr", "masked_full_psnr", "full_psnr", "full_ssim", "full_lpips",
        "mask_iou", "mask_f1", "background_opacity_mean",
        "route_shell", "route_strand", "route_residual", "checkpoint",
        "source_roots", "train_seconds", "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# Dual-input benchmark table",
        "",
        "> Rows are rank-comparable only within the same `comparable_group`.",
        "",
    ]
    for protocol in ("S-mv-official-prior", "DFA-Panda-Walk-32f-v1"):
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Setting | Method | Status | FG PSNR | Masked PSNR | Full PSNR | SSIM | LPIPS | IoU | BG alpha | Train s | Peak GB |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["protocol"] != protocol:
                continue
            lines.append(
                "| {setting} | {method} | {status} | {fg} | {masked} | {full} | {ssim} | {lpips} | {iou} | {bg} | {seconds} | {peak} |".format(
                    setting=row["setting"],
                    method=row["method"],
                    status=row["status"],
                    fg=fmt(row.get("foreground_psnr")),
                    masked=fmt(row.get("masked_full_psnr")),
                    full=fmt(row.get("full_psnr")),
                    ssim=fmt(row.get("full_ssim"), 5),
                    lpips=fmt(row.get("full_lpips"), 5),
                    iou=fmt(row.get("mask_iou"), 5),
                    bg=fmt(row.get("background_opacity_mean"), 5),
                    seconds=fmt(row.get("train_seconds"), 1),
                    peak=fmt(row.get("peak_cuda_reserved_gb"), 2),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Paired dynamic deltas: unified_soft - residual_only",
            "",
            "Positive is better for PSNR/SSIM/IoU; negative is better for LPIPS/BG alpha.",
            "",
            "| Setting | dFG PSNR | dFull PSNR | dSSIM | dLPIPS | dIoU | dBG alpha | Time x | Shell/Strand/Residual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    dynamic_rows = {
        (row["setting"], row["method"]): row
        for row in rows
        if row["protocol"] == "DFA-Panda-Walk-32f-v1" and row["status"] == "complete"
    }
    for setting in DYNAMIC_SETTINGS:
        residual = dynamic_rows.get((setting, "residual_only"))
        unified = dynamic_rows.get((setting, "unified_soft"))
        if residual is None or unified is None:
            continue

        def delta(key: str) -> float:
            return float(unified[key]) - float(residual[key])

        time_ratio = float(unified["train_seconds"]) / float(residual["train_seconds"])
        routes = "/".join(
            fmt(unified.get(f"route_{name}"), 3)
            for name in ("shell", "strand", "residual")
        )
        lines.append(
            f"| {setting} | {delta('foreground_psnr'):+.4f} | "
            f"{delta('full_psnr'):+.4f} | {delta('full_ssim'):+.5f} | "
            f"{delta('full_lpips'):+.5f} | {delta('mask_iou'):+.5f} | "
            f"{delta('background_opacity_mean'):+.5f} | {time_ratio:.2f} | {routes} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = (args.output_root or args.dynamic_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = collect(args.static_root.resolve(), args.dynamic_root.resolve())
    json_path = output_root / "dual_input_leaderboard.json"
    csv_path = output_root / "dual_input_leaderboard.csv"
    markdown_path = output_root / "dual_input_leaderboard.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
