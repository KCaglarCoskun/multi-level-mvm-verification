from multilevelmvmverification.methods import dynamic_programming_algorithm, verify_column
from multilevelmvmverification.utils.config_loader import load_config
from multilevelmvmverification.utils.timer import time_method
from multilevelmvmverification.utils.logger import export_currents_to_csv


def main():
    params = load_config('parameters.yaml')

    if params is None:
        return

    # Execute and time method
    results = {'dynamic_programming_algorithm': time_method(dynamic_programming_algorithm, params)}

    # Save results to file
    export_currents_to_csv(results['dynamic_programming_algorithm'])

    # Verification of Memristor Column
    verify_column(results['dynamic_programming_algorithm'])

if __name__ == "__main__":
    main()
