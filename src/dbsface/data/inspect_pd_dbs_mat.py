"""Smoke-test inspector for MATLAB 5 PD_DBS_Data.mat files.

This avoids a SciPy dependency for the initial data-entry check. It only parses
top-level numeric arrays and reports dimensions plus label counts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import struct
import sys
import zlib

import numpy as np


DTYPES = {
    1: "miINT8",
    2: "miUINT8",
    3: "miINT16",
    4: "miUINT16",
    5: "miINT32",
    6: "miUINT32",
    7: "miSINGLE",
    9: "miDOUBLE",
    12: "miINT64",
    13: "miUINT64",
    14: "miMATRIX",
    15: "miCOMPRESSED",
}

CLASSES = {
    1: "mxCELL",
    2: "mxSTRUCT",
    4: "mxCHAR",
    6: "mxDOUBLE",
    7: "mxSINGLE",
    8: "mxINT8",
    9: "mxUINT8",
    12: "mxINT32",
    13: "mxUINT32",
}

NUMPY_DTYPES = {
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


def pad8(n: int) -> int:
    return (n + 7) & ~7


def read_element(buf: bytes, off: int):
    if off + 8 > len(buf):
        return None
    a, b = struct.unpack_from("<II", buf, off)
    small_type = a & 0xFFFF
    small_n = (a >> 16) & 0xFFFF
    if small_n and small_type in DTYPES and small_n <= 4:
        return small_type, buf[off + 4 : off + 4 + small_n], off + 8
    start = off + 8
    end = start + b
    return a, buf[start:end], start + pad8(b)


def parse_matrix(data: bytes):
    off = 0

    flags_el = read_element(data, off)
    if not flags_el:
        return None
    _, flags_data, off = flags_el
    flags = struct.unpack_from("<II", flags_data + b"\0" * 8, 0)
    matlab_class = flags[0] & 0xFF

    dims_el = read_element(data, off)
    if not dims_el:
        return None
    dims_type, dims_data, off = dims_el
    if dims_type == 5:
        dims = list(struct.unpack("<" + "i" * (len(dims_data) // 4), dims_data))
    else:
        dims = list(struct.unpack("<" + "I" * (len(dims_data) // 4), dims_data))

    name_el = read_element(data, off)
    if not name_el:
        return None
    _, name_data, off = name_el
    name = name_data.decode("utf-8", errors="replace") if name_data else ""

    payload_type = None
    payload = b""
    if off < len(data):
        payload_el = read_element(data, off)
        if payload_el:
            payload_type, payload, _ = payload_el

    return {
        "name": name,
        "class": CLASSES.get(matlab_class, str(matlab_class)),
        "dims": dims,
        "payload_type": payload_type,
        "payload": payload,
    }


def read_variables(path: Path):
    raw = path.read_bytes()
    variables = {}
    off = 128
    while off < len(raw):
        element = read_element(raw, off)
        if not element:
            break
        dtype, data, next_off = element
        matrices = []
        if dtype == 15:
            decompressed = zlib.decompress(data)
            sub = 0
            while sub < len(decompressed):
                sub_element = read_element(decompressed, sub)
                if not sub_element:
                    break
                sub_dtype, sub_data, sub_next = sub_element
                if sub_dtype == 14:
                    matrix = parse_matrix(sub_data)
                    if matrix:
                        matrices.append(matrix)
                sub = sub_next
        elif dtype == 14:
            matrix = parse_matrix(data)
            if matrix:
                matrices.append(matrix)
        for matrix in matrices:
            variables[matrix["name"]] = matrix
        off = next_off
    return raw, variables


def matrix_to_numpy(var: dict) -> np.ndarray:
    dtype = NUMPY_DTYPES.get(var["payload_type"])
    if dtype is None:
        raise ValueError(f"Unsupported MATLAB payload type: {var['payload_type']}")
    arr = np.frombuffer(var["payload"], dtype=dtype).copy()
    return arr.reshape(tuple(var["dims"]), order="F")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/PD_DBS_Data.mat")
    raw, variables = read_variables(path)

    print(f"file: {path.resolve()}")
    print(f"bytes: {len(raw)}")
    print(f"header: {raw[:80].decode('latin1', errors='replace').rstrip()}")
    print(f"variables: {len(variables)}")

    total_images = 0
    for name in sorted(variables):
        var = variables[name]
        dims = var["dims"]
        dtype = DTYPES.get(var["payload_type"], str(var["payload_type"]))
        print(f"{name}: dims={dims}, class={var['class']}, payload={dtype}, bytes={len(var['payload'])}")
        if name.startswith("x") and dims:
            total_images += dims[0]
            if len(dims) > 1:
                print(f"  feature_count_sqrt={math.sqrt(dims[1]):.1f}")
        if name.startswith("y"):
            decoded = matrix_to_numpy(var).reshape(-1)
            counts = dict(sorted(Counter(decoded.tolist()).items()))
            print(f"  label_counts={counts}")

    has_patient_id = any("patient" in key.lower() or "id" in key.lower() for key in variables)
    print(f"total_images: {total_images}")
    print(f"patient_ids_present: {has_patient_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


