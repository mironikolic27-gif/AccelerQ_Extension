
import os
import shutil
import tempfile
import numpy as np

from kcl_util import (
    RECORD_FIELD_NAMES,
    init_or_append_csv,
    csv_length,
    label_test_extra_split,
    build_npy_index_maps,
    read_records_csv,
)
from kcl_train_xgb import train
import xgboost as xgb


def main():
    tmp = tempfile.mkdtemp(prefix="hea_csv_test_")
    data_dir = tmp

    n_rows = 10
    x_dim = 5
    x_file = "04qubits_05.X.data"
    X = np.random.randn(n_rows, x_dim)
    Y = np.random.randn(n_rows)
    np.save(os.path.join(data_dir, x_file + ".npy"), X)
    np.save(os.path.join(data_dir, x_file.replace(".X.", ".Y.") + ".npy"), Y)

    records_csv = os.path.join(tmp, "records_HEA.csv")
    for i in range(n_rows):
        status = "failed" if i == 3 else "success"
        row = {
            "record_id": i + 1,
            "ham_id": 1,
            "mining_time": 0.1,
            "true_energy": float(Y[i]) if status != "failed" else "",
            "status": status,
            "npy_index": i if status != "failed" else "",
            "x_file": x_file if status != "failed" else "",
            "peak_traced_bytes": 1000,
            "rss_before_bytes": 48_000_000,
            "rss_after_bytes": 50_000_000,
            "rss_peak_bytes": 50_000_000,
        }
        init_or_append_csv(records_csv, RECORD_FIELD_NAMES, row)

    assert csv_length(records_csv) == n_rows

    label_test_extra_split(records_csv, data_dir)
    rows = read_records_csv(records_csv)
    split_index = int(n_rows * 0.9)
    for row in rows:
        if row["status"] == "failed":
            assert row.get("split", "") != "test_extra"
            continue
        idx = int(row["npy_index"])
        if idx >= split_index:
            assert row["split"] == "test_extra", f"npy_index={idx} expected test_extra"
        else:
            assert row.get("split", "") != "test_extra"

    ham28_vec = np.ones(3)
    model_file = os.path.join(tmp, "model_test.json")
    max_size = train(
        data_dir, model_file, ham28_vec, xgb.XGBRegressor,
        0, {"n_estimators": 5, "max_depth": 3}, records_csv, params_q_size=2,
    )
    assert max_size > 0

    rows = read_records_csv(records_csv)
    train_count = sum(1 for r in rows if r.get("split") == "train")
    test_count = sum(1 for r in rows if r.get("split") == "test")
    extra_count = sum(1 for r in rows if r.get("split") == "test_extra")
    failed = [r for r in rows if r["status"] == "failed"]

    assert train_count + test_count > 0
    assert extra_count == n_rows - split_index
    assert failed[0].get("predicted_energy", "") == ""
    assert all(r.get("predicted_energy", "") != "" for r in rows if r.get("split") == "test")
    assert all(r.get("predicted_energy", "") != "" for r in rows if r.get("split") == "test_extra")

    xbig_map, xextra_map, file_meta = build_npy_index_maps(data_dir)
    assert len(xbig_map) == split_index
    assert len(xextra_map) == n_rows - split_index

    shutil.rmtree(tmp)
    print("verify_hea_csv_pipeline: OK")


if __name__ == "__main__":
    main()
