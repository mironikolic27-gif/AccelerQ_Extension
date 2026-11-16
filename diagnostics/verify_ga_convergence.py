

import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openfermion import load_operator

# kcl_util imports challenge_2024 at module load; stub it for this script only.
_challenge = types.ModuleType("utils.challenge_2024")


def _problem_hamiltonian(n_qubits, seed, hamiltonian_directory):
    return load_operator(
        file_name=f"{str(n_qubits).zfill(2)}qubits_{str(seed).zfill(2)}",
        data_directory=hamiltonian_directory,
        plain_text=False,
    )


_challenge.problem_hamiltonian = _problem_hamiltonian
sys.modules.setdefault("utils", types.ModuleType("utils"))
sys.modules["utils.challenge_2024"] = _challenge

import xgboost as xgb
from kcl_opt_xgb import opt_hyperparams
from kcl_util import ham_to_vector, load_model
from kcl_util_HEA import generate_hyper_params_hea
from kcl_util_UCCSD import generate_hyper_params_uccsd

HAM_DIR = "../hamiltonian"
DATA_HEA = "../data/HEA"
DATA_UCCSD = "../data/UCCSD"


def load_ham(prefix):
    n_qubits = int(prefix[:2])
    seed = int(prefix.split("_")[1])
    ham = load_operator(
        file_name=f"{str(n_qubits).zfill(2)}qubits_{str(seed).zfill(2)}",
        data_directory=HAM_DIR,
        plain_text=False,
    )
    return n_qubits, ham


def run_case(label, model_file, generator, prefix):
    print(f"\n{'=' * 60}\n>> Verify {label} ({prefix})\n{'=' * 60}")
    n_qubits, ham = load_ham(prefix)
    ham28_vec = ham_to_vector(ham, 50)
    max_size = load_model(model_file, xgb.XGBRegressor).n_features_in_
    _, score_ml, res, score_full, _ = opt_hyperparams(
        model_file,
        generator,
        xgb.XGBRegressor,
        n_qubits,
        ham,
        ham28_vec,
        max_size,
        None,
        None,
        None,
    )
    print(f">> {label} finished: res={'ok' if res is not None else 'None'}, "
          f"score_ml={score_ml}, score_full={score_full}")
    return res is not None


def main():
    hea_ok = run_case(
        "HEA",
        os.path.join(DATA_HEA, "model_hea_pre_xgb_24.json"),
        generate_hyper_params_hea,
        "04qubits_05",
    )
    uccsd_ok = run_case(
        "UCCSD",
        os.path.join(DATA_UCCSD, "model_uccsd_pre_xgb_24.json"),
        generate_hyper_params_uccsd,
        "08qubits_05",
    )
    if not hea_ok or not uccsd_ok:
        sys.exit(1)
    print("\n>> All convergence verification runs completed successfully.")


if __name__ == "__main__":
    main()
