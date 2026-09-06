import io
import tarfile
from pathlib import Path

from PIL import Image

from src.alignment_v3.cc3m_acquisition import (
    _classify_sample,
    _normalise_caption,
    _sample_members,
    retention_validate,
)
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_normalisation_is_deterministic() -> None:
    assert _normalise_caption(" a\t  b\n") == "a b"


def test_retention_precedes_scratch_warning_and_refreshes_all_images() -> None:
    pipeline = load_config(ROOT / "configs/cc3m_scale/pipeline.yaml")
    result = retention_validate(pipeline)
    assert result["interval_days"] == 7
    assert result["refresh_all_downloaded_tars"] is True


def test_feature_cache_is_post_training_only() -> None:
    pipeline = load_config(ROOT / "configs/cc3m_scale/pipeline.yaml")
    cache = pipeline["post_training_feature_cache"]
    assert cache["blocked_until_both_arms_complete"] is True
    assert cache["used_by_arm_a"] is False
    assert cache["used_by_arm_b"] is False


def test_training_arms_are_frozen() -> None:
    pipeline = load_config(ROOT / "configs/cc3m_scale/pipeline.yaml")
    training = pipeline["training"]
    assert training["arm_a"]["max_optimizer_steps"] == 1320
    assert training["arm_a"]["select_on_dev"] is False
    assert training["arm_b"]["max_optimizer_steps"] == 5280
    assert training["arm_b"]["early_stopping_patience_evaluations"] == 3
    assert training["seeds"] == [42, 43, 44]


def test_official_source_and_revision_policy_are_explicit() -> None:
    pipeline = load_config(ROOT / "configs/cc3m_scale/pipeline.yaml")
    acquisition = pipeline["acquisition"]
    assert acquisition["repo_id"] == "pixparse/cc3m-wds"
    assert acquisition["revision"] == "46f3d69f840e59d77d52e8decfe5baec97e94c7f"
    assert acquisition["original_cc3m_urls"] == 3318333
    assert acquisition["download_train_shards"] == 298


def test_webdataset_tar_sample_is_decoded_without_extraction(tmp_path: Path) -> None:
    image = Image.new("RGB", (512, 512), (10, 20, 30))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    tar_path = tmp_path / "sample.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in (
            ("abc.jpg", image_bytes.getvalue()),
            ("abc.txt", b"a valid caption here"),
            ("abc.url", b"https://example.invalid/image.jpg"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    samples = list(_sample_members(tar_path))
    assert len(samples) == 1
    pipeline = load_config(ROOT / "configs/cc3m_scale/pipeline.yaml")
    row = _classify_sample(
        pipeline,
        {"seed_order": 0, "filename": tar_path.name},
        samples[0][0],
        samples[0][1],
    )
    assert row["status"] == "USABLE"
    assert row["short_side"] == 512
