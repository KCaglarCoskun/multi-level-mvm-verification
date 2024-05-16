import time

import numpy as np

from multilevelmvmverification.methods import brute_force_algorithm, symmetry_based_algorithm, dynamic_programming_algorithm
from multilevelmvmverification.utils.config_loader import load_config
from multilevelmvmverification.utils.logger import setup_logger
from multilevelmvmverification.utils.results_verifier import verify_results

timing_logger = setup_logger('timing_results')
error_logger = setup_logger('error_results')


def run_analysis_for_N_w_x(N, w_max, x_max, methods_tuple, in_params):
    # Update N in the configuration
    params = in_params.copy()
    params['arch_params']['N'] = N
    params['memristor_params']['logical_max'] = w_max
    params['chip_params']['logical_max'] = x_max

    for technology_key in ['memristor_params', 'chip_params']:
        physical_noms = np.linspace(
            params[technology_key]['physical_0_nom'],
            params[technology_key]['physical_max_nom'],
            params[technology_key]['logical_max'] + 1
        )
        # physical_noms[-1] = physical_noms[-1] * 0.95  # Subtract 5% from the last value
        params[technology_key]['physical_noms'] = physical_noms

        params[technology_key]['physical_maxs'] = [
            physical_nom * (1 + params[technology_key]['relative_error'])
            for physical_nom in params[technology_key]['physical_noms']
        ]
        params[technology_key]['physical_mins'] = [
            physical_nom * (1 - params[technology_key]['relative_error'])
            for physical_nom in params[technology_key]['physical_noms']
        ]

    # f_G_nom(w_max) * f_V_nom(x_max):
    max_I_ij = params['memristor_params']['physical_noms'][-1] * params['chip_params']['physical_noms'][-1]
    max_y_ij = w_max * x_max
    f_y = lambda current: (max_y_ij/max_I_ij)*current

    # Run performance analysis and record time
    print(f"Running analysis for N={N}...")

    # Execute and time each method
    results_from_method_name = dict()
    times_from_method_name = dict()
    for i, method in enumerate(methods_tuple):
        print(f"Started Algorithm {i+1}/{len(methods_tuple)}: {method.__name__}")
        start = time.time()
        results_from_method_name[method.__name__] = method(params, f_y=f_y)
        end = time.time()
        times_from_method_name[method.__name__] = end - start

    return times_from_method_name, results_from_method_name


def main():
    params = load_config('timing_parameters.yaml')

    if params is None:
        return

    methods_from_N_w_x = {
        # (10, 3, 3): [brute_force_algorithm, symmetry_based_algorithm, dynamic_programming_algorithm],
        (10, 3, 3): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (20, 3, 3): [brute_force_algorithm, symmetry_based_algorithm, dynamic_programming_algorithm],
        # (20, 3, 3): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (40, 3, 3): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (40, 3, 3): [dynamic_programming_algorithm],
        # (80, 3, 3): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (80, 3, 3): [symmetry_based_algorithm],
        # (80, 3, 3): [dynamic_programming_algorithm],
        # (20, 5, 5): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (10, 7, 7): [symmetry_based_algorithm, dynamic_programming_algorithm],
        # (10, 7, 7): [dynamic_programming_algorithm],
        # (20, 5, 5): [symmetry_based_algorithm],
        # (20, 7, 7): [dynamic_programming_algorithm],
        # (40, 7, 7): [dynamic_programming_algorithm],
    }

    times_from_N_w_x = {}
    results_from_method_name_from_N_w_x = {}

    for N, w_max, x_max in methods_from_N_w_x:
        (
            times_from_N_w_x[(N, w_max, x_max)],
            results_from_method_name_from_N_w_x[(N, w_max, x_max)]
        ) = run_analysis_for_N_w_x(N, w_max, x_max, methods_from_N_w_x[(N, w_max, x_max)], params)
    
    # Verify results
    for _, results_from_method_name in results_from_method_name_from_N_w_x.items():
        verify_results(results_from_method_name)

    # Report timing results
    for N_w_x, times_from_method_name in times_from_N_w_x.items():
        N, w_max, x_max = N_w_x
        for method_name, time_taken in times_from_method_name.items():
            timing_logger.info(f"N={N}, w_max={w_max}, x_max={x_max}, method={method_name}, time_taken={time_taken}")
    
    # Report errors
    for N_w_x, results_from_method_name in results_from_method_name_from_N_w_x.items():
        N, w_max, x_max = N_w_x
        for method_name, results in results_from_method_name.items():
            delta_y_max = max(results['delta_y_max_min'], results['delta_y_max_max'])
            delta_y_max_rel = delta_y_max / (N * w_max * x_max)
            error_logger.info(f"N={N}, w_max={w_max}, x_max={x_max}, method={method_name},"
                              f" delta_y_max={delta_y_max}, delta_y_max_rel={delta_y_max_rel}")

if __name__ == "__main__":
    main()
