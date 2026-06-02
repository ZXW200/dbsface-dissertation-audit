"""Command dispatcher for the final submission package."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from dbsface._bootstrap import add_project_paths


COMMANDS: dict[str, list[str]] = {
    "prepare": [
        "inspect_pd_dbs_mat",
        "build_coarse_roi_masks",
    ],
    "baseline": [
        "train_baseline_mlp_numpy",
    ],
    "core-audit": [
        "check_train_test_duplicates",
        "evaluate_calibration_numpy",
        "run_roi_occlusion_mlp",
        "analyze_aev_class_statistics",
        "analyze_aev_size_normalized",
        "run_pixel_occlusion_xai_baseline",
        "run_roi_lowlevel_confound_baseline",
        "run_region_only_mlp",
    ],
    "robustness": [
        "run_near_duplicate_sensitivity",
        "run_identity_alignment_audit",
        "run_multiseed_robustness",
        "run_perturbation_robustness",
        "run_roi_supplementary_experiments",
    ],
    "advanced-audit": [
        "audit_pd_dbs_yunet_feasibility",
        "run_yunet_region_only_mlp",
        "run_yunet_roi_occlusion_mlp",
        "run_cnn_gradcam_occlusion_overlap",
        "run_cnn_yunet_roi_occlusion",
        "create_gradcam_actual_comparison_figure",
    ],
    "figures": [
        "create_dissertation_figures",
    ],
    "all": [
        "inspect_pd_dbs_mat",
        "build_coarse_roi_masks",
        "train_baseline_mlp_numpy",
        "check_train_test_duplicates",
        "evaluate_calibration_numpy",
        "run_roi_occlusion_mlp",
        "analyze_aev_class_statistics",
        "analyze_aev_size_normalized",
        "run_pixel_occlusion_xai_baseline",
        "run_roi_lowlevel_confound_baseline",
        "run_region_only_mlp",
        "run_near_duplicate_sensitivity",
        "run_identity_alignment_audit",
        "run_multiseed_robustness",
        "run_perturbation_robustness",
        "run_roi_supplementary_experiments",
        "audit_pd_dbs_yunet_feasibility",
        "run_yunet_region_only_mlp",
        "run_yunet_roi_occlusion_mlp",
        "run_cnn_gradcam_occlusion_overlap",
        "run_cnn_yunet_roi_occlusion",
        "create_gradcam_actual_comparison_figure",
        "create_dissertation_figures",
    ],
}

MODULE_IMPORTS: dict[str, str] = {
    "load_pd_dbs": "dbsface.data.load_pd_dbs",
    "inspect_pd_dbs_mat": "dbsface.data.inspect_pd_dbs_mat",
    "build_coarse_roi_masks": "dbsface.data.build_coarse_roi_masks",
    "check_train_test_duplicates": "dbsface.data.check_train_test_duplicates",
    "train_baseline_mlp_numpy": "dbsface.experiments.train_baseline_mlp_numpy",
    "evaluate_calibration_numpy": "dbsface.experiments.evaluate_calibration_numpy",
    "run_roi_occlusion_mlp": "dbsface.experiments.run_roi_occlusion_mlp",
    "analyze_aev_class_statistics": "dbsface.experiments.analyze_aev_class_statistics",
    "analyze_aev_size_normalized": "dbsface.experiments.analyze_aev_size_normalized",
    "run_pixel_occlusion_xai_baseline": "dbsface.experiments.run_pixel_occlusion_xai_baseline",
    "run_roi_lowlevel_confound_baseline": "dbsface.experiments.run_roi_lowlevel_confound_baseline",
    "run_region_only_mlp": "dbsface.experiments.run_region_only_mlp",
    "run_yunet_region_only_mlp": "dbsface.experiments.run_yunet_region_only_mlp",
    "run_yunet_roi_occlusion_mlp": "dbsface.experiments.run_yunet_roi_occlusion_mlp",
    "run_near_duplicate_sensitivity": "dbsface.robustness.run_near_duplicate_sensitivity",
    "run_identity_alignment_audit": "dbsface.robustness.run_identity_alignment_audit",
    "run_multiseed_robustness": "dbsface.robustness.run_multiseed_robustness",
    "run_perturbation_robustness": "dbsface.robustness.run_perturbation_robustness",
    "run_roi_supplementary_experiments": "dbsface.robustness.run_roi_supplementary_experiments",
    "audit_pd_dbs_yunet_feasibility": "dbsface.explain.audit_pd_dbs_yunet_feasibility",
    "run_cnn_gradcam_occlusion_overlap": "dbsface.explain.run_cnn_gradcam_occlusion_overlap",
    "run_cnn_yunet_roi_occlusion": "dbsface.explain.run_cnn_yunet_roi_occlusion",
    "create_gradcam_actual_comparison_figure": "dbsface.explain.create_gradcam_actual_comparison_figure",
    "create_dissertation_figures": "dbsface.draw.create_dissertation_figures",
    "markdown_to_latex_dissertation": "dbsface.build.markdown_to_latex_dissertation",
    "build_lancaster_dissertation": "dbsface.build.build_lancaster_dissertation",
}


def run_module(module_name: str, module_args: Sequence[str] | None = None) -> int:
    add_project_paths()
    args = list(module_args or [])
    old_argv = sys.argv[:]
    sys.argv = [module_name, *args]
    try:
        module_path = MODULE_IMPORTS.get(module_name, f"dbsface.{module_name}")
        module = importlib.import_module(module_path)
        module_main = getattr(module, "main", None)
        if module_main is None:
            raise RuntimeError(f"Module dbsface.{module_name} does not expose main().")
        result = module_main()
        return int(result or 0)
    finally:
        sys.argv = old_argv


def run_command(command: str, command_args: Sequence[str] | None = None) -> int:
    add_project_paths()
    args = list(command_args or [])
    if command == "run":
        if not args:
            raise SystemExit("Usage: py main.py run <module_name> [module arguments]")
        return run_module(args[0], args[1:])

    if command not in COMMANDS:
        valid = ", ".join(["list", "run", *COMMANDS.keys()])
        raise SystemExit(f"Unknown command '{command}'. Valid commands: {valid}")

    modules = COMMANDS[command]
    if len(modules) > 1 and args:
        raise SystemExit(
            f"Command '{command}' runs multiple modules and does not accept module arguments. "
            "Use 'py main.py run <module_name> ...' for a single module."
        )

    for module_name in modules:
        code = run_module(module_name, args if len(modules) == 1 else [])
        if code:
            return code
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the PD-DBS facial-image audit pipeline from a clean src package."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        help="Command to run: list, run, prepare, baseline, core-audit, robustness, advanced-audit, figures, all.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)

    if ns.command == "list":
        print("Available commands:")
        for name, modules in COMMANDS.items():
            print(f"  {name}: {', '.join(modules)}")
        print("  run <module_name> [args]: run one implementation module directly")
        return 0

    return run_command(ns.command, ns.args)


if __name__ == "__main__":
    raise SystemExit(main())
