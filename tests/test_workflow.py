from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import torch

from scripts.rebuild_results import parse_run_name, variant_flags
from scripts.run_seed_sweep import build_jobs, resolve_gpu_tokens
from scripts.run_finalist_comparison import build_config as build_finalist_config, normalize_gpus as normalize_finalist_gpus
from scripts.run_multitask_jobs import (
    MERGED_MARKER,
    completed_configs,
    coordinator_lock,
    normalize_gpus as normalize_multitask_gpus,
    recover_worker_outputs,
    select_checkpoint_specs,
)
from src.utils.checkpoint import load_checkpoint_metrics
from src.utils.config import load_config
from src.utils.gpu_jobs import resolve_gpu_tokens as resolve_shared_gpu_tokens


class WorkflowTests(unittest.TestCase):
    def test_nested_config_inheritance(self) -> None:
        config = load_config("configs/coco_blf_rtxpro.yaml")
        self.assertEqual(config["training"]["batch_size"], 64)
        self.assertEqual(config["data"]["num_workers"], 8)
        self.assertEqual(config["model"]["fusion_type"], "concat_mlp")

    def test_checkpoint_metrics_can_select_best_instead_of_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"metrics": {"i2t_R@1": 0.8}}, root / "best.pt")
            torch.save({"metrics": {"i2t_R@1": 0.2}}, root / "final.pt")
            self.assertEqual(load_checkpoint_metrics(root / "best.pt")["i2t_R@1"], 0.8)
            self.assertNotEqual(
                load_checkpoint_metrics(root / "best.pt")["i2t_R@1"],
                load_checkpoint_metrics(root / "final.pt")["i2t_R@1"],
            )

    def test_result_directory_parsing(self) -> None:
        self.assertEqual(
            parse_run_name("dinov2_vits14_all_minilm_l6_v2_local_global"),
            ("dinov2_vits14", "all_minilm_l6_v2", "local_global"),
        )
        self.assertEqual(variant_flags("local"), (True, False))

    def test_seed_sweep_builds_unique_jobs(self) -> None:
        sweep = {
            "seeds": [42, 43],
            "comparisons": [
                {
                    "name": "example",
                    "vision_encoder": "dinov2_vits14",
                    "text_encoder": "all_minilm_l6_v2",
                    "variants": ["baseline", "local"],
                }
            ],
        }
        jobs = build_jobs(sweep)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job["run_name"] for job in jobs}), 4)

    def test_logical_gpus_map_through_slurm_visibility(self) -> None:
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3,5,7"}, clear=False):
            self.assertEqual(resolve_gpu_tokens(["0", "1", "2"]), {"0": "3", "1": "5", "2": "7"})
            self.assertEqual(resolve_shared_gpu_tokens(["0", "1", "2"]), {"0": "3", "1": "5", "2": "7"})

    def test_all_job_runners_accept_comma_or_space_separated_gpus(self) -> None:
        self.assertEqual(normalize_finalist_gpus(["0,1", "2"]), ["0", "1", "2"])
        self.assertEqual(normalize_multitask_gpus(["0", "1,2"]), ["0", "1", "2"])

    def test_multitask_resume_does_not_treat_smoke_as_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks = ["coco_retrieval", "cifar100_zeroshot"]
            rows = []
            for config_id, mode in (("smoke_config", "smoke"), ("full_config", "full")):
                for task in tasks:
                    rows.append(
                        {
                            "config_id": config_id,
                            "task": task,
                            "status": "complete",
                            "cache": str(output / f"{config_id}__{task}__{mode}.pt"),
                        }
                    )
            pd.DataFrame(rows).to_csv(output / "evaluation_manifest.csv", index=False)
            ready = [{"task": task, "status": "ready"} for task in tasks]
            with patch("scripts.run_multitask_jobs.validate_datasets", return_value=ready):
                self.assertEqual(completed_configs(output, tasks, "full"), {"full_config"})
                self.assertEqual(completed_configs(output, tasks, "smoke"), {"smoke_config", "full_config"})

    def test_multitask_recovery_prefers_full_results_and_marks_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory) / "phase1_multitask"
            worker = canonical / ".workers" / "old_run" / "job_001"
            (canonical / "cache").mkdir(parents=True)
            (worker / "cache").mkdir(parents=True)

            base = {
                "config_id": "example",
                "vision_encoder": "convnext_tiny",
                "text_encoder": "bge_small_en",
                "variant": "baseline",
                "task": "coco_retrieval",
                "metric": "coco5_i2t_R@1",
                "best_epoch": 1,
                "checkpoint": "best.pt",
            }
            pd.DataFrame([{**base, "mode": "smoke", "value": 0.1}]).to_csv(
                canonical / "task_results_long.csv", index=False
            )
            pd.DataFrame(
                [{"config_id": "example", "task": "coco_retrieval", "mode": "smoke", "status": "complete", "cache": "example__coco__smoke.pt"}]
            ).to_csv(canonical / "evaluation_manifest.csv", index=False)
            pd.DataFrame([{"task": "coco_retrieval", "status": "ready"}]).to_csv(
                canonical / "task_status.csv", index=False
            )

            pd.DataFrame([{**base, "mode": "full", "value": 0.9}]).to_csv(
                worker / "task_results_long.csv", index=False
            )
            full_cache = worker / "cache" / "example__coco__full.pt"
            full_cache.write_bytes(b"valid-cache")
            pd.DataFrame(
                [{"config_id": "example", "task": "coco_retrieval", "mode": "full", "status": "complete", "cache": str(full_cache)}]
            ).to_csv(worker / "evaluation_manifest.csv", index=False)
            pd.DataFrame([{"task": "coco_retrieval", "status": "ready"}]).to_csv(
                worker / "task_status.csv", index=False
            )

            self.assertEqual(recover_worker_outputs(canonical), 1)
            merged = pd.read_csv(canonical / "task_results_long.csv")
            self.assertEqual(merged.iloc[0]["mode"], "full")
            self.assertEqual(merged.iloc[0]["value"], 0.9)
            self.assertTrue((canonical / "cache" / full_cache.name).exists())
            self.assertTrue((worker / MERGED_MARKER).exists())
            self.assertEqual(recover_worker_outputs(canonical), 0)

    def test_multitask_coordinator_lock_rejects_a_second_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory)
            with coordinator_lock(canonical):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with coordinator_lock(canonical):
                        pass

    def test_multitask_explicit_blf_ids_do_not_expand_to_full_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            requested = [
                "dinov2_vits14__all_minilm_l6_v2__local",
                "convnextv2_tiny__all_minilm_l6_v2__local_global",
                "dinov2_vits14__bge_small_en__local",
            ]
            directories = [
                "dinov2_vits14_all_minilm_l6_v2_local",
                "convnextv2_tiny_all_minilm_l6_v2_local_global",
                "dinov2_vits14_bge_small_en_local",
                "efficientnet_b0_distilbert_baseline",
                "swin_tiny_e5_small_v2_global",
            ]
            for name in directories:
                path = checkpoint_root / name
                path.mkdir()
                (path / "best.pt").touch()

            specs = select_checkpoint_specs(
                checkpoint_root, "baseline_plus_reliable_blf", requested
            )
            self.assertEqual({spec.config_id for spec in specs}, set(requested))

    def test_finalist_worker_builds_frozen_comparable_job_config(self) -> None:
        base = load_config("configs/coco_blf_rtxpro.yaml")
        job = {
            "use_local_blf": False, "use_global_blf": False,
            "save_dir": "checkpoints/example", "variant": "baseline",
        }
        config = build_finalist_config(base, job, 15)
        self.assertEqual(config["training"]["epochs"], 15)
        self.assertEqual(config["model"]["vision_encoder"], "convnext_tiny")
        self.assertEqual(config["model"]["text_encoder"], "minilm_l6")

    def test_notebook_contains_rerunnable_seed_sweep_section(self) -> None:
        notebook = json.loads(Path("notebooks/01_experiment_workflow.ipynb").read_text())
        cell_ids = {cell.get("id") for cell in notebook["cells"]}
        self.assertIn("launch-seed-sweep-on-allocated-node", cell_ids)
        self.assertIn("launch-seed-sweep-directly", cell_ids)
        self.assertIn("batch-seed-sweep-results", cell_ids)
        self.assertIn("batch-seed-sweep-analysis", cell_ids)

    def test_training_markdown_documents_gpu_tradeoffs(self) -> None:
        notebook = json.loads(Path("notebooks/01_experiment_workflow.ipynb").read_text())
        by_id = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
        for cell_id in ("step-2-md", "step-4-md", "step-6-md", "launch-seed-sweep-on-allocated-node"):
            source = by_id[cell_id]
            for count in ("| 1 |", "| 2 |", "| 4 |", "| 8 |"):
                self.assertIn(count, source)
        seed_code = by_id["launch-seed-sweep-directly"]
        self.assertIn("SEED_SWEEP_GPU_IDS = list(GPU_IDS)", seed_code)
        self.assertNotIn("range(2)", seed_code)

    def test_phase15_notebook_cells_execute_real_pipeline_functions(self) -> None:
        notebook = json.loads(Path("notebooks/01_experiment_workflow.ipynb").read_text())
        by_id = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
        self.assertIn("profile_phase15_efficiency(", by_id["step-15-efficiency"])
        self.assertIn("run_hard_negative_mining(", by_id["step-16-hard-negatives"])
        self.assertIn("scripts/run_training_hard_negative_jobs.py", by_id["step-16-hard-negatives"])
        self.assertIn('"--gpus", *GPU_IDS', by_id["step-16-hard-negatives"])
        self.assertNotIn('device="cuda:0"', by_id["step-16-hard-negatives"])
        self.assertIn("run_expert_selection(", by_id["step-17-selection"])
        self.assertIn("requires a GPU allocation", by_id["step-15-efficiency"])


if __name__ == "__main__":
    unittest.main()
