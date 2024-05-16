import os
import sys
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from methods import (brute_force_algorithm,
                     symmetry_based_algorithm,
                     dynamic_programming_algorithm,
                     dynamic_programming_algorithm_with_abstraction)
from utils.logger import setup_logger
from utils.results_verifier import verify_results

error_logger = setup_logger('error_results')


def set_params(G_max, V_max, x_max, w_max):
    params = dict()
    params['arch_params'] = dict()
    params['arch_params']['N'] = 6
    params['chip_params'] = dict()
    params['chip_params']['logical_max'] = x_max
    physical_noms = V_max*np.sort(np.random.rand(x_max+1))
    negative_error_factor = 1 - np.random.rand(x_max+1)
    positive_error_factor = 1 + np.random.rand(x_max+1)
    params['chip_params']['physical_mins'] = np.sort(physical_noms*negative_error_factor)
    params['chip_params']['physical_maxs'] = np.sort(physical_noms*positive_error_factor)

    params['memristor_params'] = dict()
    params['memristor_params']['logical_max'] = w_max
    physical_noms = G_max*np.sort(np.random.rand(w_max+1))
    negative_error_factor = 1 - np.random.rand(w_max+1)
    positive_error_factor = 1 + np.random.rand(w_max+1)
    params['memristor_params']['physical_mins'] = np.sort(physical_noms*negative_error_factor)
    params['memristor_params']['physical_maxs'] = np.sort(physical_noms*positive_error_factor)
    return params


def get_f_y(w_max, x_max, params):
    max_I_ij = (params['memristor_params']['physical_maxs'][-1]
                * params['chip_params']['physical_maxs'][-1])  # f_G_max(w_max) * f_V_max(x_max):
    max_y_ij = w_max * x_max
    f_y = lambda current: (max_y_ij/max_I_ij)*current
    return f_y


def main():
    for i in range(10):
        print(f"Running test {i+1}/10...")
        N = 10
        w_max_max = 7
        x_max_max = 7
        G_max = 1e-3  # 1 mS
        V_max = 5  # 5 V

        x_max = np.random.randint(1, x_max_max)
        w_max = np.random.randint(1, w_max_max)

        params = set_params(G_max, V_max, x_max, w_max)

        if N*w_max*x_max < 50:
            methods = (brute_force_algorithm,
                       symmetry_based_algorithm,
                       dynamic_programming_algorithm,
                       dynamic_programming_algorithm_with_abstraction)
        else:
            methods = (symmetry_based_algorithm,
                       dynamic_programming_algorithm,
                       dynamic_programming_algorithm_with_abstraction)

        # results_from_method_name_list = {}

        f_y = get_f_y(w_max, x_max, params)

        # Execute each method
        results_from_method_name = dict()
        for i, method in enumerate(methods):
            print(f"Started Algorithm {i+1}/{len(methods)}: {method.__name__} for test {i+1}/10...")
            results_from_method_name[method.__name__] = method(params, f_y=f_y)

        # Verify results
        # for _, results_from_method_name in results_from_method_name_from_N_w_x.items():
        verify_results(results_from_method_name)

        # Report errors
        # for results_from_method_name in results_from_method_name_from_N_w_x.items():
        for method_name, results in results_from_method_name.items():
            delta_y_max = max(results['delta_y_max_min'], results['delta_y_max_max'])
            delta_y_max_rel = delta_y_max / (N * w_max * x_max)
            error_logger.info(f"N={N}, w_max={w_max}, x_max={x_max}, method={method_name},"
                              f" delta_y_max={delta_y_max}, delta_y_max_rel={delta_y_max_rel}")


if __name__ == "__main__":
    main()
