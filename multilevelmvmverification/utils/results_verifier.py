import math
import numpy as np
from multilevelmvmverification.utils.logger import setup_logger

method_equivalence_logger = setup_logger('method_equivalence_results')


def are_results_equivalent(result1, result2):
    """
    Compares the results from two formal multilevel MVM verification methods and
    returns True if the results are equivalent, False otherwise.
    """
    numeric_results = {'delta_y_max_min', 'delta_y_max_max'}
    numeric_results_per_y = {'I_min_from_y', 'I_max_from_y'}
    # w_x_star_min and w_x_star_max results can be different even for correct
    # methods, especially if the V and R values are linearly distributed.
    # Therefore, we don't check them anymore.
    #
    # tuple_of_vectors_results = {'w_x_star_min_from_y', 'w_x_star_max_from_y'}
    tuple_of_vectors_results = set()
    results_with_y_keys = {'I_min_from_y', 'I_max_from_y'}

    common_numeric_results = numeric_results & result1.keys() & result2.keys()
    common_numeric_results_per_y = numeric_results_per_y & result1.keys() & result2.keys()
    common_tuple_of_vectors_results = tuple_of_vectors_results & result1.keys() & result2.keys()
    common_results_with_y_keys = results_with_y_keys & result1.keys() & result2.keys()

    # Compare y keys.
    S_y_some = result1['I_min_from_y'].keys()
    for key_name in common_results_with_y_keys:
        for i, result in enumerate((result1, result2)):
            if result[key_name].keys() != S_y_some:
                method_equivalence_logger.info(
                    "Results aren't equivalent since the y values are different"
                    f"between result 1, key I_min_from_y, and result {i+1}, key {key_name}."
                )
                return False

    for numeric_result in common_numeric_results:
        if not math.isclose(result1[numeric_result], result2[numeric_result]):
            method_equivalence_logger.info(
                f"Results aren't equivalent since the {numeric_result} values are different: "
                f"{result1[numeric_result]} vs {result2[numeric_result]}"
            )
            return False
    for y in S_y_some:
        for numeric_result in common_numeric_results_per_y:
            if not math.isclose(result1[numeric_result][y], result2[numeric_result][y]):
                method_equivalence_logger.info(
                    f"Results aren't equivalent since the {numeric_result} values are different for "
                    f"y = {y}: {result1[numeric_result][y]} vs {result2[numeric_result][y]}"
                )
                return False
        for tuple_of_vectors_result in common_tuple_of_vectors_results:
            X_1_sorted = np.sort(result1[tuple_of_vectors_result][y][0])
            X_2_sorted = np.sort(result2[tuple_of_vectors_result][y][0])
            M_1_sorted = np.sort(result1[tuple_of_vectors_result][y][1])
            M_2_sorted = np.sort(result2[tuple_of_vectors_result][y][1])
            if ((X_1_sorted != X_2_sorted).any() or (M_1_sorted != M_2_sorted).any()):
                method_equivalence_logger.info(
                    f"Results aren't equivalent since the {tuple_of_vectors_result} values are different for "
                    f"y = {y}"
                )
                return False
    return True


def verify_results(results_from_method_name):
    """
    Compares the results from all formal multilevel MVM verification methods and
    writes the results to a log file called method_equivalence_results.log. Also
    returns True if the results are equivalent, False otherwise.
    """
    method_name_list = list(results_from_method_name.keys())
    oracle_method_name = method_name_list[0]
    for i in range(1, len(method_name_list)):
        if not are_results_equivalent(results_from_method_name[oracle_method_name],
                                      results_from_method_name[method_name_list[i]]):
            method_equivalence_logger.info(
                f"!Warning: Methods {oracle_method_name} and {method_name_list[i]} produce different results."
            )
            return False
        method_equivalence_logger.info(
            f"Methods {oracle_method_name} and {method_name_list[i]} produce equivalent results."
        )
    method_equivalence_logger.info("All methods produce equivalent results.")
    return True
