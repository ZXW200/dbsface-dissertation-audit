"""PD-DBS MATLAB data loader.

The local data file is a MATLAB 5 `.mat` file with `x_train`, `x_test`,
`y_train`, and `y_test`. SciPy is not required for this loader; it uses the
lightweight MATLAB parser packaged with `dbsface`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from dbsface.inspect_pd_dbs_mat import read_variables
except ImportError:  # pragma: no cover - allows direct script execution
    from dbsface.data.inspect_pd_dbs_mat import read_variables


DTYPE_MAP = {
    1: np.dtype("<i1"),
    2: np.dtype("u1"),
    3: np.dtype("<i2"),
    4: np.dtype("<u2"),
    5: np.dtype("<i4"),
    6: np.dtype("<u4"),
    7: np.dtype("<f4"),
    9: np.dtype("<f8"),
    12: np.dtype("<i8"),
    13: np.dtype("<u8"),
}


def matrix_to_numpy(var: dict) -> np.ndarray:
    dtype = DTYPE_MAP.get(var["payload_type"])
    if dtype is None:
        raise ValueError(f"Unsupported MATLAB payload type: {var['payload_type']}")
    arr = np.frombuffer(var["payload"], dtype=dtype).copy()
    return arr.reshape(tuple(var["dims"]), order="F")


def flat_to_images(x: np.ndarray) -> np.ndarray:
    """Convert [N, 1024] rows to [N, 32, 32, 1] images.

    The public Face2Gene notebook displays each row with
    `row.reshape((32, 32)).T`; keep that orientation for all downstream code.
    """

    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"Expected [N, 1024] data, got {x.shape}")
    images = x.reshape(x.shape[0], 32, 32).transpose(0, 2, 1)
    return images[..., None].astype(np.float32)


def load_pd_dbs(path: str | Path = "data/raw/PD_DBS_Data.mat") -> dict[str, np.ndarray]:
    _, variables = read_variables(Path(path))
    required = {"x_train", "x_test", "y_train", "y_test"}
    missing = required.difference(variables)
    if missing:
        raise KeyError(f"Missing required variables: {sorted(missing)}")

    x_train_flat = matrix_to_numpy(variables["x_train"]).astype(np.float32)
    x_test_flat = matrix_to_numpy(variables["x_test"]).astype(np.float32)
    y_train = matrix_to_numpy(variables["y_train"]).reshape(-1).astype(np.int64)
    y_test = matrix_to_numpy(variables["y_test"]).reshape(-1).astype(np.int64)

    if x_train_flat.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train length mismatch")
    if x_test_flat.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test length mismatch")

    return {
        "x_train_flat": x_train_flat,
        "x_test_flat": x_test_flat,
        "x_train_images": flat_to_images(x_train_flat),
        "x_test_images": flat_to_images(x_test_flat),
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> int:
    data = load_pd_dbs(sys.argv[1] if len(sys.argv) > 1 else "data/raw/PD_DBS_Data.mat")
    for key, value in data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




