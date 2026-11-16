"""
Smoke test for kcl_util_HEA integration. Run from src/ with AccelerQ env:

    python3.10 smoke_hea_integration.py

"""

import numpy as np
from kcl_util import process_file
from kcl_util_HEA import (
    generate_hyper_params_hea,
    wrapper_hea,
    compress_hea,
    sanitize_x_vec_params,
    default_hea_x_vec,
)
from kcl_HEA_VQE import Wrapper_HEA, Solver_HEA, Ansatz_HEA
from kcl_util_hamiltonian import load_hamiltonian, compress_hamiltonian, to_qubit_operator
from kcl_tests_HEA import test_static_hea, test_semi_dynamic_hea


def main():
    n = 4
    x = generate_hyper_params_hea(1, n)
    w = Wrapper_HEA(n, None, True, 3, 0, 1, 0, 0, 1000, 1e-5, 1e-6, 10**4, 0, 0, 5)
    assert np.allclose(x, w._build_x_vec()), "x_vec parity failed"
    print("x_vec parity OK")

    bad = x.copy()
    bad[6] = 0
    bad[11] = 2
    assert round(sanitize_x_vec_params(bad)[11]) == 0
    print("sanitize COBYLA OK")

    assert test_static_hea(x, None) and test_semi_dynamic_hea(x, None)
    print("kcl_tests_HEA OK")

    r = process_file("../hamiltonian/", "04qubits_05.data")
    ham = compress_hea(r[1])
    n_qubits = int(r[0])
    print(f"compress OK: {len(ham.terms)} terms")

    y = wrapper_hea(x, n_qubits, ham)
    print(f"wrapper_hea energy OK: {y}")

    ham0 = load_hamiltonian(n_qubits, 0, "../hamiltonian")
    ham0 = compress_hamiltonian(ham0, x[8])
    op = to_qubit_operator(ham0)
    ans = Ansatz_HEA(n_qubits, 3, 0, 1, 0)
    Solver_HEA(
        op, ans.circuit, True, 0, 1000, 1e-6, 10**4, 0, 0, 5, x, n_qubits
    )
    print("Solver_HEA construct OK")

    assert len(default_hea_x_vec(n)) == 14
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
