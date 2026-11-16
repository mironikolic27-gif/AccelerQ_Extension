"""
Run from src/:

    python3 verify_stage1_profiling.py
"""

from kcl_util import (
    mining_profile_csv_fields,
    profile_mining_call,
    MINING_PROFILE_FIELD_NAMES,
)


def main():
    result, prof, err = profile_mining_call(lambda: 42)
    assert err is None
    assert result == 42
    assert prof["peak_traced_bytes"] is not None

    fields = mining_profile_csv_fields(prof)
    assert set(fields.keys()) == set(MINING_PROFILE_FIELD_NAMES)
    assert fields["peak_traced_bytes"] != ""

    _, prof_fail, err_fail = profile_mining_call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert isinstance(err_fail, ValueError)
    assert prof_fail["peak_traced_bytes"] is not None

    empty = mining_profile_csv_fields({})
    assert all(empty[k] == "" for k in MINING_PROFILE_FIELD_NAMES)

    print("All Stage 1 profiling checks passed.")


if __name__ == "__main__":
    main()
