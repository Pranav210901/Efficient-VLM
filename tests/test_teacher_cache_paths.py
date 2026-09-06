from pathlib import Path

import torch

from src.alignment_v3.distillation import TeacherCache, _key_from_normalised_path


def test_lookup_survives_dataset_move_behind_symlink(tmp_path: Path) -> None:
    physical_root = tmp_path / "quarantine" / "coco"
    physical_root.mkdir(parents=True)
    logical_root = tmp_path / "data" / "coco"
    logical_root.parent.mkdir()
    logical_root.symlink_to(physical_root, target_is_directory=True)
    logical_image = logical_root / "train.jpg"
    physical_image = physical_root / "train.jpg"
    physical_image.touch()
    caption = "a test image"

    cache = TeacherCache.__new__(TeacherCache)
    resolved_image = str(physical_image.resolve())
    cache.image_index = {resolved_image: 0}
    cache.cached_image_paths = {resolved_image: str(logical_image.absolute())}
    cache.text_index = {
        _key_from_normalised_path(str(logical_image.absolute()), caption): 0
    }
    cache.image_embeddings = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    cache.text_embeddings = torch.tensor([[3.0, 4.0]], dtype=torch.float16)

    images, texts = cache.lookup(
        [str(logical_image)], [str(logical_image)], [caption], torch.device("cpu")
    )

    torch.testing.assert_close(images, cache.image_embeddings)
    torch.testing.assert_close(texts, cache.text_embeddings)
