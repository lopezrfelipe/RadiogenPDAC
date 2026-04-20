from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from radiogenpdac.config import load_all_configs
from radiogenpdac.framework import load_yaml, render_framework_summary, write_summary
from radiogenpdac.manifests import (
    COHORT_REQUIRED_COLUMNS,
    GENOMICS_REQUIRED_COLUMNS,
    merge_manifests,
    validate_manifest,
)
from radiogenpdac.splits import SplitConfig, build_split_table

app = typer.Typer(add_completion=False, help="PDAC radiogenomics framework utilities.")
console = Console()


@app.command("validate-manifest")
def validate_manifest_command(
    cohort: Path = typer.Option(..., exists=True, readable=True, help="Path to cohort CSV."),
    genomics: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Optional genomics CSV.",
    ),
) -> None:
    issues = validate_manifest(cohort, COHORT_REQUIRED_COLUMNS)
    if genomics:
        issues.extend(validate_manifest(genomics, GENOMICS_REQUIRED_COLUMNS))

    if issues:
        for issue in issues:
            console.print(f"[red]- {issue}[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Manifest validation passed.[/green]")


@app.command("merge-manifests")
def merge_manifests_command(
    cohort: Path = typer.Option(..., exists=True, readable=True),
    genomics: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Path to merged CSV."),
    join_key: str = typer.Option("patient_id", help="Shared key column."),
) -> None:
    merged = merge_manifests(cohort, genomics, output, join_key=join_key)
    console.print(f"[green]Merged {len(merged)} rows into {output}.[/green]")


@app.command("make-splits")
def make_splits_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Long-form split assignment CSV."),
    n_folds: int = typer.Option(5, min=2),
    group_column: str = typer.Option("patient_id"),
    stratify_column: str | None = typer.Option(None),
    test_fraction: float = typer.Option(0.15, min=0.0, max=0.5),
    seed: int = typer.Option(2026),
) -> None:
    import pandas as pd

    frame = pd.read_csv(manifest)
    split_table = build_split_table(
        frame,
        SplitConfig(
            n_folds=n_folds,
            group_column=group_column,
            stratify_column=stratify_column,
            test_fraction=test_fraction,
            seed=seed,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(output, index=False)
    console.print(f"[green]Wrote split assignments to {output}.[/green]")


@app.command("render-plan")
def render_plan_command(
    data_config: Path = typer.Option(..., exists=True, readable=True),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    target_config: Path = typer.Option(..., exists=True, readable=True),
    train_config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Path to framework JSON summary."),
) -> None:
    payload = render_framework_summary(
        data_cfg=load_yaml(data_config),
        model_cfg=load_yaml(model_config),
        target_cfg=load_yaml(target_config),
        train_cfg=load_yaml(train_config),
    )
    write_summary(output, payload)
    console.print(f"[green]Framework plan written to {output}.[/green]")


@app.command("preprocess-cohort")
def preprocess_cohort_command(
    manifest: Path = typer.Option(..., exists=True, readable=True, help="Merged manifest CSV."),
    data_config: Path = typer.Option(..., exists=True, readable=True),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    output_dir: Path = typer.Option(..., help="Directory for patient token files."),
    output_manifest: Path = typer.Option(..., help="CSV manifest with preprocessed file paths."),
) -> None:
    from radiogenpdac.preprocessing import preprocess_manifest

    processed = preprocess_manifest(
        manifest_path=manifest,
        output_dir=output_dir,
        data_cfg=load_yaml(data_config),
        model_cfg=load_yaml(model_config),
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_manifest, index=False)
    console.print(f"[green]Preprocessed {len(processed)} cases into {output_manifest}.[/green]")


@app.command("train")
def train_command(
    manifest: Path = typer.Option(..., exists=True, readable=True, help="Preprocessed manifest CSV."),
    splits: Path = typer.Option(..., exists=True, readable=True, help="Split assignment CSV."),
    fold: int = typer.Option(0, help="Fold to train."),
    data_config: Path = typer.Option(..., exists=True, readable=True),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    target_config: Path = typer.Option(..., exists=True, readable=True),
    train_config: Path = typer.Option(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("artifacts/runs/fold0"), help="Training output directory."),
) -> None:
    from radiogenpdac.train import run_training

    configs = load_all_configs(data_config, model_config, target_config, train_config)
    checkpoint = run_training(
        manifest_path=manifest,
        split_path=splits,
        fold=fold,
        data_cfg=configs["data"],
        model_cfg=configs["model"],
        target_cfg=configs["target"],
        train_cfg=configs["train"],
        output_dir=output_dir,
    )
    console.print(f"[green]Training complete. Best checkpoint: {checkpoint}[/green]")


@app.command("export-embeddings")
def export_embeddings_command(
    checkpoint: Path = typer.Option(..., exists=True, readable=True),
    manifest: Path = typer.Option(..., exists=True, readable=True),
    splits: Path = typer.Option(..., exists=True, readable=True),
    fold: int = typer.Option(0),
    data_config: Path = typer.Option(..., exists=True, readable=True),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    target_config: Path = typer.Option(..., exists=True, readable=True),
    train_config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Output JSON file for embeddings."),
) -> None:
    from radiogenpdac.train import export_embeddings

    configs = load_all_configs(data_config, model_config, target_config, train_config)
    destination = export_embeddings(
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        split_path=splits,
        fold=fold,
        data_cfg=configs["data"],
        model_cfg=configs["model"],
        target_cfg=configs["target"],
        train_cfg=configs["train"],
        output_path=output,
    )
    console.print(f"[green]Embeddings written to {destination}.[/green]")


@app.command("prepare-encoder-dataset")
def prepare_encoder_dataset_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    phase: str = typer.Option(..., help="Imaging phase to prepare, such as venous or arterial."),
    dataset_id: int = typer.Option(..., help="nnU-Net dataset ID."),
    dataset_name: str = typer.Option(..., help="nnU-Net dataset name suffix."),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(..., help="Path to nnUNet_raw."),
    output_index: Path = typer.Option(..., help="CSV index of prepared training cases."),
) -> None:
    from radiogenpdac.pdac_encoder import prepare_phase_finetune_dataset

    prepared = prepare_phase_finetune_dataset(
        manifest_path=manifest,
        phase=phase,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        output_index_path=output_index,
    )
    console.print(f"[green]Prepared {len(prepared)} {phase} cases for nnU-Net fine-tuning.[/green]")


@app.command("plan-encoder-dataset")
def plan_encoder_dataset_command(
    dataset_id: int = typer.Option(...),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    nnunet_results_dir: Path = typer.Option(...),
    plans_identifier: str = typer.Option("nnUNetPlans"),
    configurations: str = typer.Option("3d_fullres", help="Comma-separated nnU-Net configurations."),
    gpu_memory_target_gb: float = typer.Option(8.0),
    num_processes: int = typer.Option(4),
    verify_integrity: bool = typer.Option(False),
    overwrite_target_spacing: str | None = typer.Option(
        None,
        help="Optional comma-separated spacing in z,y,x order, for example 1.5,0.8,0.8",
    ),
) -> None:
    from radiogenpdac.pdac_encoder import plan_and_preprocess_phase_dataset

    spacing = (
        [float(item.strip()) for item in overwrite_target_spacing.split(",")]
        if overwrite_target_spacing
        else None
    )
    plan_and_preprocess_phase_dataset(
        dataset_id=dataset_id,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        plans_identifier=plans_identifier,
        configurations=[item.strip() for item in configurations.split(",") if item.strip()],
        gpu_memory_target_gb=gpu_memory_target_gb,
        num_processes=num_processes,
        verify_integrity=verify_integrity,
        overwrite_target_spacing=spacing,
    )
    console.print("[green]Finished nnU-Net planning and preprocessing.[/green]")


@app.command("finetune-encoder")
def finetune_encoder_command(
    dataset_id: int = typer.Option(...),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    nnunet_results_dir: Path = typer.Option(...),
    pretrained_weights: Path = typer.Option(..., exists=True, readable=True),
    configuration: str = typer.Option("3d_fullres"),
    fold: int = typer.Option(0),
    trainer_name: str = typer.Option("nnUNetTrainer_ftce"),
    plans_identifier: str = typer.Option("nnUNetPlans"),
    device: str = typer.Option("cuda"),
    num_gpus: int = typer.Option(1),
) -> None:
    from radiogenpdac.pdac_encoder import finetune_phase_encoder

    model_dir = finetune_phase_encoder(
        dataset_id=dataset_id,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_name,
        plans_identifier=plans_identifier,
        pretrained_weights=pretrained_weights,
        device=device,
        num_gpus=num_gpus,
    )
    console.print(f"[green]Fine-tuning complete. Model directory: {model_dir}[/green]")


@app.command("extract-encoder-features")
def extract_encoder_features_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    phase: str = typer.Option(...),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    nnunet_results_dir: Path = typer.Option(...),
    model_training_output_dir: Path = typer.Option(..., exists=True, readable=True),
    fold: int = typer.Option(0),
    output_dir: Path = typer.Option(..., help="Directory for per-patient feature vectors."),
    output_manifest: Path = typer.Option(..., help="CSV with patient_id to feature path mappings."),
    checkpoint_name: str = typer.Option("checkpoint_final.pth"),
    device: str = typer.Option("cuda"),
    intensity_window_hu: str = typer.Option("-100,240"),
    margin_mm: str = typer.Option("20,20,20"),
) -> None:
    from radiogenpdac.pdac_encoder import extract_phase_encoder_features

    features = extract_phase_encoder_features(
        manifest_path=manifest,
        phase=phase,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        model_training_output_dir=model_training_output_dir,
        fold=fold,
        output_dir=output_dir,
        output_manifest=output_manifest,
        checkpoint_name=checkpoint_name,
        device=device,
        intensity_window_hu=[float(item.strip()) for item in intensity_window_hu.split(",")],
        margin_mm=[float(item.strip()) for item in margin_mm.split(",")],
    )
    console.print(f"[green]Extracted {len(features)} {phase} encoder feature vectors.[/green]")


@app.command("attach-encoder-features")
def attach_encoder_features_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    feature_manifest: Path = typer.Option(..., exists=True, readable=True),
    phase: str = typer.Option(...),
    output_manifest: Path = typer.Option(..., help="Updated manifest including encoder feature paths."),
) -> None:
    from radiogenpdac.pdac_encoder import attach_phase_encoder_features

    merged = attach_phase_encoder_features(
        manifest_path=manifest,
        feature_manifest_path=feature_manifest,
        phase=phase,
        output_manifest=output_manifest,
    )
    console.print(f"[green]Wrote updated manifest with {phase} encoder features for {len(merged)} rows.[/green]")


@app.command("build-phase-ingestion")
def build_phase_ingestion_command(
    input_csv: Path = typer.Option(..., exists=True, readable=True, help="Long-form phase CSV."),
    output_csv: Path = typer.Option(..., help="Resolved phase manifest CSV."),
    structure_patterns_json: str | None = typer.Option(
        None,
        help="Optional JSON map from structure name to filename keywords.",
    ),
) -> None:
    from radiogenpdac.ingestion import DEFAULT_STRUCTURE_PATTERNS, build_phase_ingestion_manifest

    if structure_patterns_json:
        import json

        parsed = json.loads(structure_patterns_json)
        patterns = {str(key): [str(item) for item in value] for key, value in parsed.items()}
    else:
        patterns = DEFAULT_STRUCTURE_PATTERNS

    frame = build_phase_ingestion_manifest(
        input_csv=input_csv,
        output_csv=output_csv,
        structure_patterns=patterns,
    )
    console.print(f"[green]Resolved masks for {len(frame)} phase rows into {output_csv}.[/green]")


@app.command("discover-cluster-phase-manifest")
def discover_cluster_phase_manifest_command(
    framework_root: Path = typer.Option(
        Path("."),
        exists=True,
        file_okay=False,
        readable=True,
        help="Framework repo root on the cluster.",
    ),
    output_csv: Path = typer.Option(..., help="Output CSV for discovered patient-phase rows."),
    data_root: Path | None = typer.Option(
        None,
        help="Optional explicit data root. Defaults to ../data relative to framework_root.",
    ),
    phases: str = typer.Option("venous,arterial", help="Comma-separated phases to scan."),
) -> None:
    from radiogenpdac.ingestion import discover_cluster_phase_manifest

    manifest = discover_cluster_phase_manifest(
        framework_root=framework_root,
        output_csv=output_csv,
        data_root=data_root,
        phases=[item.strip() for item in phases.split(",") if item.strip()],
    )
    console.print(f"[green]Discovered {len(manifest)} cluster patient-phase rows into {output_csv}.[/green]")


@app.command("scan-cluster-complete-cases")
def scan_cluster_complete_cases_command(
    framework_root: Path = typer.Option(
        Path("."),
        exists=True,
        file_okay=False,
        readable=True,
        help="Framework repo root on the cluster.",
    ),
    output_dir: Path = typer.Option(..., help="Directory for discovered and filtered manifests."),
    data_root: Path | None = typer.Option(
        None,
        help="Optional explicit data root. Defaults to ../data relative to framework_root.",
    ),
    phases: str = typer.Option("venous,arterial", help="Comma-separated phases to scan."),
    required_structures: str = typer.Option(
        "tumor,pancreas,duct,cbd,artery,vein",
        help="Comma-separated structures required for a case to be marked complete.",
    ),
    structure_patterns_json: str | None = typer.Option(
        None,
        help="Optional JSON map from structure name to filename keywords.",
    ),
) -> None:
    from radiogenpdac.ingestion import (
        DEFAULT_STRUCTURE_PATTERNS,
        scan_cluster_complete_cases,
    )

    if structure_patterns_json:
        import json

        parsed = json.loads(structure_patterns_json)
        structure_patterns = {str(key): [str(item) for item in value] for key, value in parsed.items()}
    else:
        structure_patterns = DEFAULT_STRUCTURE_PATTERNS

    outputs = scan_cluster_complete_cases(
        framework_root=framework_root,
        output_dir=output_dir,
        data_root=data_root,
        phases=[item.strip() for item in phases.split(",") if item.strip()],
        structure_patterns=structure_patterns,
        required_structures=[item.strip() for item in required_structures.split(",") if item.strip()],
    )
    venous_path = outputs.get("venous")
    arterial_path = outputs.get("arterial")
    console.print(
        "[green]Cluster scan complete.[/green] "
        f"Inventory: {outputs['inventory']} "
        f"Venous manifest: {venous_path} "
        f"Arterial manifest: {arterial_path}"
    )


@app.command("build-cohort-from-phases")
def build_cohort_from_phases_command(
    phase_manifest: Path = typer.Option(..., exists=True, readable=True),
    output_csv: Path = typer.Option(..., help="Wide cohort manifest CSV."),
) -> None:
    from radiogenpdac.ingestion import build_wide_cohort_manifest_from_phase_table

    cohort = build_wide_cohort_manifest_from_phase_table(
        phase_manifest_csv=phase_manifest,
        output_csv=output_csv,
    )
    console.print(f"[green]Built wide cohort manifest with {len(cohort)} patients.[/green]")


@app.command("prepare-ingested-encoder-dataset")
def prepare_ingested_encoder_dataset_command(
    phase_manifest: Path = typer.Option(..., exists=True, readable=True),
    phase: str = typer.Option(...),
    dataset_id: int = typer.Option(...),
    dataset_name: str = typer.Option(...),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    output_index: Path = typer.Option(...),
    task_mode: str = typer.Option(
        "multiclass",
        help="multiclass or tumor_only. Multiclass is the recommended default.",
    ),
    crop_mode: str = typer.Option(
        "pancreas_roi",
        help="pancreas_roi, tumor_roi, or none. pancreas_roi best matches the crop-first detector.",
    ),
    crop_margin_mm: str = typer.Option(
        "80,80,30",
        help="Crop margin in mm as x,y,z around the crop structure.",
    ),
    structure_priority: str = typer.Option(
        "pancreas,duct,cbd,artery,vein,tumor",
        help="Comma-separated merge priority for multiclass labels.",
    ),
    label_map_json: str | None = typer.Option(
        None,
        help="Optional JSON map from structure name to integer label.",
    ),
) -> None:
    from radiogenpdac.ingestion import DEFAULT_LABEL_MAP, prepare_phase_finetune_dataset_from_ingestion

    if label_map_json:
        import json

        label_map = {str(key): int(value) for key, value in json.loads(label_map_json).items()}
    else:
        label_map = DEFAULT_LABEL_MAP

    prepared = prepare_phase_finetune_dataset_from_ingestion(
        phase_manifest_csv=phase_manifest,
        phase=phase,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        output_index_csv=output_index,
        task_mode=task_mode,
        structure_priority=[item.strip() for item in structure_priority.split(",") if item.strip()],
        label_map=label_map,
        crop_mode=crop_mode,
        crop_margin_mm=[float(item.strip()) for item in crop_margin_mm.split(",") if item.strip()],
    )
    console.print(f"[green]Prepared {len(prepared)} ingested {phase} cases for {task_mode} fine-tuning.[/green]")


@app.command("write-encoder-splits")
def write_encoder_splits_command(
    prepared_index: Path = typer.Option(..., exists=True, readable=True),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    dataset_id: int = typer.Option(...),
    dataset_name: str = typer.Option(...),
    output_json: Path | None = typer.Option(None),
    split_column: str | None = typer.Option(
        None,
        help="Optional column with train/val assignments from the ingestion CSV.",
    ),
    n_folds: int = typer.Option(5, min=2),
    seed: int = typer.Option(12345),
) -> None:
    from radiogenpdac.ingestion import write_nnunet_splits

    destination = write_nnunet_splits(
        prepared_index_csv=prepared_index,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        output_json=output_json,
        split_column=split_column,
        n_folds=n_folds,
        seed=seed,
    )
    console.print(f"[green]Wrote nnU-Net splits to {destination}.[/green]")


@app.command("evaluate-encoder-model")
def evaluate_encoder_model_command(
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    nnunet_results_dir: Path = typer.Option(...),
    model_training_output_dir: Path = typer.Option(..., exists=True, readable=True),
    images_folder: Path = typer.Option(..., exists=True, readable=True),
    reference_folder: Path = typer.Option(..., exists=True, readable=True),
    output_folder: Path = typer.Option(...),
    reference_tumor_label: int = typer.Option(1),
    prediction_tumor_label: int = typer.Option(1),
    split_json: Path | None = typer.Option(None, exists=True, readable=True),
    fold: int = typer.Option(0),
    checkpoint_name: str = typer.Option("checkpoint_final.pth"),
    device: str = typer.Option("cuda"),
) -> None:
    from radiogenpdac.ingestion import evaluate_encoder_model_on_split

    summary = evaluate_encoder_model_on_split(
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        model_training_output_dir=model_training_output_dir,
        images_folder=images_folder,
        reference_folder=reference_folder,
        split_json=split_json,
        fold=fold,
        output_folder=output_folder,
        reference_tumor_label=reference_tumor_label,
        prediction_tumor_label=prediction_tumor_label,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    console.print(
        f"[green]Evaluation complete. Mean Dice={summary['mean_dice']:.4f}, "
        f"mean tumor GT coverage={summary['mean_tumor_gt_coverage']:.4f}[/green]"
    )


@app.command("summarize-validation-tumor-metrics")
def summarize_validation_tumor_metrics_command(
    reference_folder: Path = typer.Option(..., exists=True, readable=True),
    prediction_folder: Path = typer.Option(..., exists=True, readable=True),
    output_json: Path = typer.Option(...),
    reference_tumor_label: int = typer.Option(1),
    prediction_tumor_label: int = typer.Option(1),
) -> None:
    from radiogenpdac.ingestion import compute_tumor_metrics_on_folder

    summary = compute_tumor_metrics_on_folder(
        reference_folder=reference_folder,
        prediction_folder=prediction_folder,
        case_ids=None,
        reference_tumor_label=reference_tumor_label,
        prediction_tumor_label=prediction_tumor_label,
        output_json=output_json,
    )
    console.print(
        f"[green]Validation tumor metrics complete. Mean Dice={summary['mean_dice']:.4f}, "
        f"mean tumor GT coverage={summary['mean_tumor_gt_coverage']:.4f}[/green]"
    )


@app.command("run-phase-finetune-workflow")
def run_phase_finetune_workflow_command(
    phase_manifest: Path = typer.Option(..., exists=True, readable=True),
    phase: str = typer.Option(...),
    dataset_id: int = typer.Option(...),
    dataset_name: str = typer.Option(...),
    workflow_root: Path = typer.Option(..., help="Directory for workflow outputs."),
    pdac_root: Path = typer.Option(Path("PDAC_Detection"), exists=True, file_okay=False, readable=True),
    nnunet_raw_dir: Path = typer.Option(...),
    nnunet_preprocessed_dir: Path = typer.Option(...),
    nnunet_results_dir: Path = typer.Option(...),
    pretrained_weights: Path = typer.Option(..., exists=True, readable=True),
    original_model_training_output_dir: Path = typer.Option(..., exists=True, readable=True),
    task_mode: str = typer.Option("multiclass"),
    crop_mode: str = typer.Option(
        "pancreas_roi",
        help="pancreas_roi, tumor_roi, or none. pancreas_roi is recommended for crop-first fine-tuning.",
    ),
    crop_margin_mm: str = typer.Option("80,80,30"),
    split_column: str | None = typer.Option(
        "split",
        help="Optional split column from the ingestion CSV. Use empty string to disable.",
    ),
    plans_identifier: str = typer.Option("nnUNetPlans"),
    configuration: str = typer.Option("3d_fullres"),
    trainer_name: str = typer.Option("nnUNetTrainer_ftce"),
    device: str = typer.Option("cuda"),
    num_gpus: int = typer.Option(1),
    fold: int = typer.Option(0),
    n_folds: int = typer.Option(5, min=2),
    seed: int = typer.Option(12345),
    gpu_memory_target_gb: float = typer.Option(8.0),
    num_processes: int = typer.Option(4),
    overwrite_target_spacing: str | None = typer.Option(None),
    structure_priority: str = typer.Option("pancreas,duct,cbd,artery,vein,tumor"),
    label_map_json: str | None = typer.Option(None),
    checkpoint_name: str = typer.Option("checkpoint_final.pth"),
) -> None:
    from radiogenpdac.ingestion import DEFAULT_LABEL_MAP, run_phase_finetune_workflow

    if label_map_json:
        import json

        label_map = {str(key): int(value) for key, value in json.loads(label_map_json).items()}
    else:
        label_map = DEFAULT_LABEL_MAP

    spacing = (
        [float(item.strip()) for item in overwrite_target_spacing.split(",")]
        if overwrite_target_spacing
        else None
    )
    summary = run_phase_finetune_workflow(
        phase_manifest_csv=phase_manifest,
        phase=phase,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        pretrained_weights=pretrained_weights,
        original_model_training_output_dir=original_model_training_output_dir,
        workflow_root=workflow_root,
        task_mode=task_mode,
        crop_mode=crop_mode,
        crop_margin_mm=[float(item.strip()) for item in crop_margin_mm.split(",") if item.strip()],
        split_column=split_column or None,
        plans_identifier=plans_identifier,
        configuration=configuration,
        trainer_name=trainer_name,
        device=device,
        num_gpus=num_gpus,
        n_folds=n_folds,
        seed=seed,
        structure_priority=[item.strip() for item in structure_priority.split(",") if item.strip()],
        label_map=label_map,
        gpu_memory_target_gb=gpu_memory_target_gb,
        num_processes=num_processes,
        overwrite_target_spacing=spacing,
        checkpoint_name=checkpoint_name,
        fold=fold,
    )
    console.print(
        f"[green]Workflow complete. Baseline Dice={summary['baseline_mean_dice']:.4f}, "
        f"post-finetune Dice={summary['post_finetune_mean_dice']:.4f}. "
        f"Summary: {summary['workflow_summary_json']}[/green]"
    )


if __name__ == "__main__":
    app()
