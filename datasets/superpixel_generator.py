import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans

DEFAULT_SUPERPIXEL_ROOT_ENV = "SET_SUPERPIXEL_ROOT"


def _resolve_output_root(output_root: Optional[Union[str, Path]]) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser()

    env_root = os.environ.get(DEFAULT_SUPERPIXEL_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser()

    raise ValueError(
        "output_root is required (set via argument or the SET_SUPERPIXEL_ROOT environment variable)"
    )


def format_sub_sp(raw: np.ndarray, sp: np.ndarray, sp_num: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sp_merge = np.zeros_like(sp, dtype=np.int32)
    sp_max = np.amax(sp)
    sub_sp = np.zeros([sp_max + 1, 5], dtype=np.int64)
    for i in range(sp_max + 1):
        sp_index = np.where(sp == i)
        sub_sp[i][0] = np.mean(sp_index[0])
        sub_sp[i][1] = np.mean(sp_index[1])
        sub_sp[i][2] = np.mean(raw[sp_index][:, 0])
        sub_sp[i][3] = np.mean(raw[sp_index][:, 1])
        sub_sp[i][4] = np.mean(raw[sp_index][:, 2])

    kmeans = KMeans(n_clusters=k, random_state=436, n_init="auto")
    kmeans.fit_predict(sub_sp)
    sp_index = kmeans.labels_
    reformat_index = np.zeros_like(sp_index, dtype=np.int32) - 1

    value = -1
    idx = -1
    for i in range(len(sp_index)):
        if sp_index[i] != value and reformat_index[i] == -1:
            idx += 1
            value = sp_index[i]
            reformat_index[np.where(sp_index == value)] = idx

    for i in range(sp_max + 1):
        sp_merge[np.where(sp == i)] = reformat_index[i]

    return reformat_index, sp_merge


def save_graph(
    image_path: Union[str, Path],
    seed: int,
    *,
    output_root: Optional[Union[str, Path]] = None,
    size: int = 256,
    sp_num: int = 1024,
    save_name: str = "T_sp",
) -> None:
    """
    Generate multi-stage superpixel graphs for a single image and persist them as .npy files.

    Args:
        image_path: Path to the RGB dermoscopy image.
        seed: Controls slic compactness and the output folder grouping.
        output_root: Root directory where `{size}_{sp_num}/{save_name}/{seed}/stage*/` folders are created.
                     Defaults to the path specified by the SET_SUPERPIXEL_ROOT environment variable.
        size: Target square resolution used for SLIC.
        sp_num: Number of superpixels for the first stage.
        save_name: Sub-directory name (defaults to "T_sp").
    """

    output_root = _resolve_output_root(output_root)
    image_path = Path(image_path)
    mask_name = image_path.stem

    base_dir = output_root / f"{size}_{sp_num}"
    stage_dir = base_dir / save_name / str(seed)
    v2_dir = base_dir / f"{save_name}_v2" / str(seed)
    preview_dir = base_dir / f"{save_name}_preview" / str(seed)

    for root in (stage_dir, v2_dir, preview_dir):
        for stage in range(1, 6):
            (root / f"stage{stage}").mkdir(parents=True, exist_ok=True)

    stage_paths = [stage_dir / f"stage{i}" / f"{mask_name}.npy" for i in range(1, 6)]
    stage_paths_v2 = [v2_dir / f"stage{i}" / f"{mask_name}.npy" for i in range(1, 6)]

    stage1_exists = stage_paths[0].exists()
    if stage1_exists:
        for src, dst in zip(stage_paths, stage_paths_v2):
            data = np.load(src)
            np.save(dst, np.int32(data))
        return

    raw = cv2.imread(str(image_path))
    if raw is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    raw = raw[2:-2, 2:-2, :]
    raw = cv2.resize(raw, (size, size))
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    sp = slic(raw, n_segments=sp_num, compactness=seed, start_label=0, max_num_iter=10)

    index, sub_sp = format_sub_sp(raw, sp, sp_num, sp_num // 4)
    index2, sub_sp2 = format_sub_sp(raw, sub_sp, sp_num // 4, sp_num // 16)
    index3, sub_sp3 = format_sub_sp(raw, sub_sp2, sp_num // 16, sp_num // 64)
    index4, sub_sp4 = format_sub_sp(raw, sub_sp3, sp_num // 64, sp_num // 256)

    np.save(stage_paths[0], np.int32(sp))
    np.save(stage_paths[1], np.int32(index))
    np.save(stage_paths[2], np.int32(index2))
    np.save(stage_paths[3], np.int32(index3))
    np.save(stage_paths[4], np.int32(index4))


__all__ = ["format_sub_sp", "save_graph"]

