

from kcl_HEA_VQE import Ansatz_HEA, _build_hea_circuit
from kcl_UCCSD_VQE import Ansatz_UCCSD, _build_uccsd_circuit
from kcl_util_ansatz_cache import (
    INVALID_NUM_PARAMS,
    cached_hea_num_params,
    cached_uccsd_num_params,
)
from kcl_util_HEA import default_hea_x_vec
from kcl_util_UCCSD import default_uccsd_x_vec, generate_hyper_params_uccsd
from kcl_tests_HEA import test_semi_dynamic_hea
from kcl_test_UCCSD import test_semi_dynamic_uccsd


def _assert_cache_hit(cached_fn, *args):
    cached_fn.cache_clear()
    first = cached_fn(*args)
    second = cached_fn(*args)
    info = cached_fn.cache_info()
    assert first == second
    assert info.hits == 1, f"expected 1 cache hit, got {info}"
    assert info.misses == 1, f"expected 1 cache miss, got {info}"


def main():
    _assert_cache_hit(cached_hea_num_params, 4, 2, 0, 1, 0)
    print("HEA num_params cache hit OK")

    _assert_cache_hit(cached_uccsd_num_params, 4, 0, 2, 0, 0)
    print("UCCSD num_params cache hit OK")

    _assert_cache_hit(_build_hea_circuit, 4, 2, 0, 1, 0)
    print("HEA circuit cache hit OK")

    _assert_cache_hit(_build_uccsd_circuit, 4, 0, 2, 0, 0)
    print("UCCSD circuit cache hit OK")

    hea_a = Ansatz_HEA(4, 2, 0, 1, 0)
    hea_b = Ansatz_HEA(4, 2, 0, 1, 0)
    assert id(hea_a.circuit) != id(hea_b.circuit)
    assert hea_a.num_parameters == hea_b.num_parameters
    print("HEA circuit copy independence OK")

    uccsd_a = Ansatz_UCCSD(4, 0, 2, 0, 0)
    uccsd_b = Ansatz_UCCSD(4, 0, 2, 0, 0)
    assert id(uccsd_a.circuit) != id(uccsd_b.circuit)
    assert uccsd_a.num_parameters == uccsd_b.num_parameters
    print("UCCSD circuit copy independence OK")

    _build_uccsd_circuit.cache_clear()
    for seed in range(1, 21):
        x = generate_hyper_params_uccsd(seed, 4)
        Ansatz_UCCSD(
            4,
            int(round(x[2])),
            int(round(x[3])),
            int(round(x[4])),
            int(round(x[5])),
        )
    mining_info = _build_uccsd_circuit.cache_info()
    print("UCCSD mining simulation cache stats:", mining_info)
    assert mining_info.hits > mining_info.misses, (
        f"expected hits > misses in mining simulation, got {mining_info}"
    )

    hea_x = default_hea_x_vec(4)
    assert test_semi_dynamic_hea(hea_x, None)
    print("test_semi_dynamic_hea OK")

    uccsd_x = default_uccsd_x_vec(4)
    assert test_semi_dynamic_uccsd(uccsd_x, None)
    print("test_semi_dynamic_uccsd OK")

    cached_hea_num_params.cache_clear()
    invalid = cached_hea_num_params(4, -1, 0, 1, 0)
    assert invalid == INVALID_NUM_PARAMS
    print("invalid combo sentinel OK")

    print("All ansatz cache checks passed.")


if __name__ == "__main__":
    main()
