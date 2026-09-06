from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from scripts.run_alignment_v3_notebook_cell import selected_cell_indices
from src.alignment_v3.distillation import symmetric_kl_distillation
from src.alignment_v3.fingerprint import (
    Fingerprint,
    hash_config,
    hash_dataset_split,
    is_fresh,
    write_fingerprint,
)
from src.alignment_v3.model import LoRALinear, SpatialTokenAdapter
from src.alignment_v3.runner import (
    _training_stage_jobs,
    ablation_jobs,
    build_job_config,
    final_jobs,
    pair_jobs,
    recovery_confirmation_jobs,
    select_recovery_batches,
    select_recovery_pair,
    select_recovery_recipe,
    select_pair,
    sensitivity_jobs,
    transfer_jobs,
)
from src.alignment_v3.splits import create_development_split
from src.alignment_v3.training import _link_epoch_snapshot
from src.utils.config import load_config


class AlignmentV3Tests(unittest.TestCase):
    def test_epoch_snapshot_is_immutable_across_latest_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "latest.pt"
            torch.save({"epoch": 1, "value": "first"}, latest)
            epoch_one = _link_epoch_snapshot(latest, 1)
            replacement = root / "latest.pt.tmp"
            torch.save({"epoch": 2, "value": "second"}, replacement)
            replacement.replace(latest)
            epoch_two = _link_epoch_snapshot(latest, 2)
            self.assertEqual(torch.load(epoch_one, weights_only=False)["value"], "first")
            self.assertEqual(torch.load(epoch_two, weights_only=False)["value"], "second")
            self.assertNotEqual(epoch_one.stat().st_ino, epoch_two.stat().st_ino)

    def test_config_hash_is_independent_of_dict_insertion_order(self) -> None:
        self.assertEqual(hash_config({"a": 1, "b": {"c": 2}}), hash_config({"b": {"c": 2}, "a": 1}))

    def test_dataset_hash_is_set_order_independent_but_rejects_duplicates(self) -> None:
        self.assertEqual(hash_dataset_split(["b", "a"]), hash_dataset_split(["a", "b"]))
        with self.assertRaises(ValueError):
            hash_dataset_split(["a", "a"])

    def test_fingerprint_stale_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fingerprint.json"
            value = Fingerprint("a", "b", "c", "d", "e", "f", seed=42)
            write_fingerprint(path, value)
            self.assertTrue(is_fresh(path, value))
            self.assertFalse(is_fresh(path, Fingerprint("changed", "b", "c", "d", "e", "f", seed=42)))

    def test_development_split_is_disjoint_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"image_path": f"train/{image}.jpg", "caption": f"caption {caption}"}
                for image in range(10)
                for caption in range(2)
            ]
            final = [{"image_path": f"val/{image}.jpg", "caption": "final"} for image in range(3)]
            pd.DataFrame(rows).to_csv(root / "source.csv", index=False)
            pd.DataFrame(final).to_csv(root / "final.csv", index=False)
            first = create_development_split(
                root / "source.csv", root / "train.csv", root / "dev.csv", root / "manifest.json",
                dev_images=3, seed=7, final_csv=root / "final.csv",
            )
            second = create_development_split(
                root / "source.csv", root / "train2.csv", root / "dev2.csv", root / "manifest2.json",
                dev_images=3, seed=7, final_csv=root / "final.csv",
            )
            self.assertEqual(first["train_split_hash"], second["train_split_hash"])
            self.assertEqual(first["dev_split_hash"], second["dev_split_hash"])
            self.assertEqual(first["overlap_counts"], {"train_dev": 0, "train_final": 0, "dev_final": 0})

    def test_spatial_adapter_handles_vit_and_convnext_grids(self) -> None:
        adapter = SpatialTokenAdapter(32, 32, adapter_dim=16, heads=4, ff_dim=32, dropout=0)
        for shape in ((14, 14), (7, 7)):
            tokens = torch.randn(2, shape[0] * shape[1], 32)
            output = adapter(tokens, torch.randn(2, 32), shape)
            self.assertEqual(output.shape, (2, 32))

    def test_lora_starts_as_exact_base_mapping(self) -> None:
        base = nn.Linear(8, 6)
        lora = LoRALinear(base, rank=2, alpha=4, dropout=0)
        values = torch.randn(3, 8)
        self.assertTrue(torch.equal(lora(values), base(values)))
        self.assertFalse(lora.lora_b.weight.detach().bool().any())

    def test_realistic_distillation_kl_is_finite(self) -> None:
        tensors = [torch.randn(4, 16), torch.randn(8, 16), torch.randn(4, 16), torch.randn(8, 16)]
        loss = symmetric_kl_distillation(*tensors, temperature=2, student_scale=14.3, teacher_scale=20)
        self.assertTrue(torch.isfinite(loss))

    def test_pipeline_expansion_matches_compute_plan(self) -> None:
        pipeline = load_config("configs/alignment_v3/pipeline.yaml")
        self.assertEqual(len(pair_jobs(pipeline)), 6)
        self.assertEqual(len(sensitivity_jobs(pipeline)), 8)
        self.assertEqual(len(ablation_jobs(pipeline)), 14)
        self.assertEqual(len(final_jobs(pipeline)), 3)
        self.assertEqual(len(transfer_jobs(pipeline)), 6)

    def test_pair_config_uses_native_resolution_and_scaled_large_batch_lr(self) -> None:
        pipeline = load_config("configs/alignment_v3/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            # Keep the unit test independent of any real completed batch probe.
            pipeline["output_root"] = directory
            jobs = pair_jobs(pipeline)
            vit_job = next(job for job in jobs if job["vision_encoder"] == "dinov3_vits16")
            config = build_job_config(pipeline, vit_job, "pair")
        self.assertEqual(config["data"]["image_size"], 256)
        self.assertEqual(config["training"]["batch_size"], 768)
        self.assertGreater(config["training"]["lr"], 0.0003)
        self.assertLessEqual(config["training"]["applied_lr_scale"], 3.0)

    def test_recovery_sweep_is_locked_and_uses_fixed_batches_with_scaled_lr(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        jobs = pair_jobs(pipeline)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job["vision_encoder"] for job in jobs}, {"dinov3_vits16"})
        self.assertEqual({job["text_encoder"] for job in jobs}, {"all_minilm_l6_v2"})
        self.assertEqual({job["seed"] for job in jobs}, {42})
        self.assertEqual(
            [job["batch_size"] for job in jobs],
            [512, 768, 1024, 2048],
        )
        for job in jobs:
            config = build_job_config(pipeline, job, "pair")
            batch_size = job["batch_size"]
            expected_scale = min(math.sqrt(batch_size / 128), 3.0)
            self.assertEqual(config["training"]["batch_size"], batch_size)
            self.assertAlmostEqual(config["training"]["applied_lr_scale"], expected_scale)
            self.assertAlmostEqual(config["training"]["lr"], 0.0003 * expected_scale)

    def test_pair_stage_does_not_require_future_recovery_shortlist(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with patch(
            "src.alignment_v3.runner.recovery_confirmation_jobs",
            side_effect=AssertionError("future shortlist was resolved eagerly"),
        ):
            jobs = _training_stage_jobs(pipeline, "pair")
        self.assertEqual(len(jobs), 4)

    def test_recovery_confirmation_uses_only_shortlisted_batches_and_seed_43(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            pipeline["output_root"] = directory
            selection = Path(directory) / "selection"
            selection.mkdir()
            (selection / "batch_shortlist.json").write_text(
                json.dumps({"batch_sizes": [768, 512]})
            )
            jobs = recovery_confirmation_jobs(pipeline)
        self.assertEqual([job["batch_size"] for job in jobs], [768, 512])
        self.assertEqual({job["seed"] for job in jobs}, {43})
        self.assertEqual(len(jobs), 2)

    def test_recovery_preserves_official_pair_gate(self) -> None:
        official = load_config("configs/alignment_v3/pipeline.yaml")["selection"]
        recovery = load_config("configs/alignment_v3_recovery/pipeline.yaml")["selection"]
        for key in (
            "pair_minimum_mean_r1",
            "pair_minimum_gain_over_v2",
            "v2_best_mean_r1",
        ):
            self.assertEqual(recovery[key], official[key])

    def test_recovery_adaptive_confirmation_and_gate(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            pipeline["output_root"] = directory
            pipeline_path = Path(directory) / "pipeline.yaml"
            import yaml

            pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False))
            scores = {512: 0.19, 768: 0.21, 1024: 0.205, 2048: 0.17}
            for job in pair_jobs(pipeline):
                destination = Path(directory) / "pair" / job["run_id"]
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_text(
                    json.dumps(
                        {
                            "seed": 42,
                            "mean_R@1": scores[job["batch_size"]],
                            "bidirectional_pair_latency_ms": 3.8,
                            "params_total_inference": 45_000_000,
                            "params_trainable_inference": 1_500_000,
                        }
                    )
                )
            shortlist = select_recovery_batches(pipeline_path)
            self.assertEqual(shortlist["batch_sizes"], [768, 1024])
            for job in recovery_confirmation_jobs(pipeline):
                destination = Path(directory) / "confirmation" / job["run_id"]
                destination.mkdir(parents=True)
                confirmation_score = 0.205 if job["batch_size"] == 768 else 0.20
                (destination / "metrics.json").write_text(
                    json.dumps(
                        {
                            "seed": 43,
                            "mean_R@1": confirmation_score,
                            "bidirectional_pair_latency_ms": 3.8,
                            "params_total_inference": 45_000_000,
                            "params_trainable_inference": 1_500_000,
                        }
                    )
                )
            selected = select_recovery_pair(pipeline_path)
        self.assertTrue(selected["proceed"])
        self.assertEqual(selected["batch_size"], 768)
        self.assertEqual(selected["seeds"], [42, 43])

    def test_standard_pair_selector_records_recovery_batch(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            pipeline["output_root"] = directory
            pipeline_path = Path(directory) / "pipeline.yaml"
            import yaml

            pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False))
            scores = {512: 0.18, 768: 0.205, 1024: 0.19, 2048: 0.17}
            for job in pair_jobs(pipeline):
                destination = Path(directory) / "pair" / job["run_id"]
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_text(
                    json.dumps(
                        {
                            "run_id": job["run_id"],
                            "experiment_id": job["experiment_id"],
                            "mean_R@1": scores[job["batch_size"]],
                            "bidirectional_pair_latency_ms": 3.8,
                            "params_total_inference": 45_000_000,
                        }
                    )
                )
            selected = select_pair(pipeline_path)
        self.assertTrue(selected["proceed"])
        self.assertEqual(selected["experiment_id"], "dinov3_vits16__all_minilm_l6_v2")
        self.assertEqual(selected["batch_size"], 768)

    def test_selected_recovery_batch_propagates_to_recipe_with_scaled_lr(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            pipeline["output_root"] = directory
            selection = Path(directory) / "selection"
            selection.mkdir()
            (selection / "pair.json").write_text(
                json.dumps(
                    {
                        "proceed": True,
                        "vision_encoder": "dinov3_vits16",
                        "text_encoder": "all_minilm_l6_v2",
                        "batch_size": 512,
                    }
                )
            )
            job = next(
                value for value in ablation_jobs(pipeline)
                if value["experiment_id"] == "baseline" and value["seed"] == 42
            )
            config = build_job_config(pipeline, job, "ablation")
        self.assertEqual(config["training"]["batch_size"], 512)
        self.assertAlmostEqual(config["training"]["lr"], 0.0006)

    def test_recovery_recipe_requires_two_seed_gain_and_budgets(self) -> None:
        pipeline = load_config("configs/alignment_v3_recovery/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            pipeline["output_root"] = directory
            pipeline_path = Path(directory) / "pipeline.yaml"
            import yaml

            pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False))
            selection = Path(directory) / "selection"
            selection.mkdir()
            (selection / "pair.json").write_text(json.dumps({"proceed": True}))
            scores = {
                "baseline": {42: 0.20, 43: 0.21},
                "distillation": {42: 0.225, 43: 0.235},
                "lora": {42: 0.21, 43: 0.22},
                "distillation_lora": {42: 0.23, 43: 0.24},
            }
            for job in ablation_jobs(pipeline):
                destination = Path(directory) / "ablation" / job["run_id"]
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_text(
                    json.dumps(
                        {
                            "experiment_id": job["experiment_id"],
                            "seed": job["seed"],
                            "mean_R@1": scores[job["experiment_id"]][job["seed"]],
                            "bidirectional_pair_latency_ms": 4.0,
                            "params_total_inference": 60_000_000,
                            "params_trainable_inference": 4_000_000,
                        }
                    )
                )
            selected = select_recovery_recipe(pipeline_path)
        self.assertTrue(selected["proceed"])
        self.assertEqual(selected["experiment_id"], "distillation_lora")
        self.assertGreaterEqual(
            selected["minimum_seed_gain_over_matched_baseline"], 0.02
        )

    def test_result_notebook_maps_every_command(self) -> None:
        notebook = json.loads(Path("notebooks/03_alignment_v3_results.ipynb").read_text())
        commands = (
            "validate", "prefetch", "reference", "batch-probe", "train-pair", "eval-pair", "select-pair",
            "teacher-cache", "train-sensitivity", "eval-sensitivity", "select-distillation",
            "component-smoke", "train-ablation", "eval-ablation", "select-recipe",
            "train-final", "eval-final", "transfer", "oracle", "report", "status",
        )
        for command in commands:
            indices = selected_cell_indices(notebook, command)
            self.assertGreaterEqual(len(indices), 2, command)
            self.assertEqual(indices[0], 1)


if __name__ == "__main__":
    unittest.main()
