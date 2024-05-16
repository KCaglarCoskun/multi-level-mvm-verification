import time

import numpy as np

from multilevelmvmverification.utils.console_printer import print_progress_bar
from multilevelmvmverification.utils.logger import setup_logger


def get_point_from_single_index(
    index: int,
    n_dimensions: int,
    n_points_per_dimension: int
):
    """
    This function assumes an "n_dimensions" dimensional point grid where each
    dimension has "n_points_per_dimension" number of points and returns the
    point corresponding to the given index. The index is assumed to be in the
    range [0,n_points_per_dimension^n_dimensions - 1].
    """
    point = np.zeros(n_dimensions, dtype=int)
    for i in range(n_dimensions):
        point[n_dimensions - i - 1] = index % n_points_per_dimension
        index //= n_points_per_dimension
    return point


def get_all_y_values_N_eq_1(w_max, x_max):
    """
    This function returns the set of all possible y_i values for the given w_max
    and x_max. It implements the equation given in the report with label:
    eq:set_of_Y_values for N = 1. It is then also equal to the equation given in
    the report with label: eq:set_of_p_j_values.
    """
    S_y_1 = set()
    for x_j in range(x_max + 1):
        for w_j in range(w_max + 1):
            S_y_1.add(x_j*w_j)
    return S_y_1


def abstracting_function(params: dict):
    """
    This function abstracts w_ij and x_j with p_j such that p_j = w_ij * x_j. It
    then returns the minimum/maximum possible current for the given p_j (the
    dictionaries f_i_min_pj/f_i_max_pj) and also the w_ij and x_j values that
    minimize/maximize the current (the dictionaries wj_xj_star_max_from_pj and
    wj_xj_star_min_from_pj).
    """
    # Extract necessary parameters
    x_max = params['chip_params']['logical_max']
    V_mins = params['chip_params']['physical_mins']
    V_maxs = params['chip_params']['physical_maxs']
    w_max = params['memristor_params']['logical_max']
    G_mins = params['memristor_params']['physical_mins']
    G_maxs = params['memristor_params']['physical_maxs']

    # Initialize dictionaries
    f_i_max_pj = dict()
    wj_xj_star_max_from_pj = dict()
    f_i_min_pj = dict()
    wj_xj_star_min_from_pj = dict()

    # Iterate over all possible w_ij and x_j values. Computes f_i_max_pj
    # according to equation (eq:max_current_for_p_j). Also computes
    # wj_xj_star_min_from_pj and wj_xj_star_max_from_pj in the meantime.
    for w_j in range(w_max + 1):
        for x_j in range(x_max + 1):
            p_j = w_j * x_j
            f_i_min_wj_xj = G_mins[w_j]*V_mins[x_j]
            f_i_max_wj_xj = G_maxs[w_j]*V_maxs[x_j]
            if f_i_min_pj.get(p_j, np.inf) > f_i_min_wj_xj:
                f_i_min_pj[p_j] = f_i_min_wj_xj
                wj_xj_star_min_from_pj[p_j] = (w_j, x_j)
            if f_i_max_pj.get(p_j, 0) < f_i_max_wj_xj:
                f_i_max_pj[p_j] = f_i_max_wj_xj
                wj_xj_star_max_from_pj[p_j] = (w_j, x_j)
    f_i_min_pj, f_i_max_pj, wj_xj_star_min_from_pj, wj_xj_star_max_from_pj
    return f_i_min_pj, f_i_max_pj, wj_xj_star_min_from_pj, wj_xj_star_max_from_pj


def get_w_x_star(p_star_from_y, wj_xj_star_from_pj, S_y):
    """
    This function returns the \vec{w} and \vec{x} vectors that minimize/maximize
    the current for each y value in S_y. It implements the algorithm given in
    the report with label: alg:get_vec_XM_max_from_y.
    """
    N = len(p_star_from_y[0])
    w_and_x_from_y = dict()
    for y in S_y:  # RLine 2
        p_star = p_star_from_y[y]  # RLine 3
        w_star = np.zeros(N, dtype=int)
        x_star = np.zeros(N, dtype=int)
        for i in range(N):  # RLine 4
            w_star[i], x_star[i] = wj_xj_star_from_pj[p_star[i]]  # RLine 5
        w_and_x_from_y[y] = (w_star, x_star)  # RLine 6
    return w_and_x_from_y  # RLine 7


def brute_force_algorithm(params: dict, f_y: callable = None):
    """
    This function implements the naive algorithm. It is given in the report
    (label: alg:naive_algorithm) only for finding the maximum values. Therefore,
    the line numbers (RLine) are given only for the maximum values.
    """
    # Extract necessary parameters
    N = params['arch_params']['N']
    x_max = params['chip_params']['logical_max']
    w_max = params['memristor_params']['logical_max']

    # Start Algorithm
    S_y_1 = get_all_y_values_N_eq_1(w_max, x_max)  # RLine 1
    p_j_list_sorted = sorted(S_y_1)
    (f_i_min_pj,
     f_i_max_pj,
     wj_xj_star_min_from_pj,
     wj_xj_star_max_from_pj) = abstracting_function(params)  # RLine 2
    f_I_min = np.full((max(S_y_1)*N + 1, N), np.inf)
    f_I_max = np.zeros((max(S_y_1)*N + 1, N))  # RLine 3
    p_star_min_from_y = dict()
    p_star_max_from_y = dict()  # RLine 4
    S_y = set()
    total_iterations = len(S_y_1)**N
    prev_percent = -1  # For progress bar
    start_time = time.time()  # For approximate remaining time calculation
    for i_vec_p in range(total_iterations):  # RLine 5.1
        # Note that S_p_j = S_y_1
        vec_p_state_point = get_point_from_single_index(i_vec_p, N, len(S_y_1))  # RLine 5.2
        # vec_p_state_point has elements in the range [0, len(S_p_j) - 1] whereas
        # vec_p has elements from S_p_j which are in the range [0, max(S_p_j)] but
        # have some holes. Therefore, we map the numbers to the S_p_j values, below.
        vec_p = [p_j_list_sorted[p_j_state_num] for p_j_state_num in vec_p_state_point]  # RLine 5.3
        y = sum(vec_p)  # RLine 6
        S_y.add(y)
        min_current_from_vec_p = sum([f_i_min_pj[p_j] for p_j in vec_p])
        max_current_from_vec_p = sum([f_i_max_pj[p_j] for p_j in vec_p])
        if min_current_from_vec_p < f_I_min[y][N - 1]:
            f_I_min[y][N - 1] = min_current_from_vec_p
            p_star_min_from_y[y] = vec_p
        if max_current_from_vec_p > f_I_max[y][N - 1]:  # RLine 7
            f_I_max[y][N - 1] = max_current_from_vec_p  # RLine 8
            p_star_max_from_y[y] = vec_p  # RLine 9
        # Print progress bar
        prev_percent = print_progress_bar(i_vec_p + 1, total_iterations, start_time, prev_percent=prev_percent)
    if f_y:
        (delta_y_max_min,
         delta_y_max_max,
         y_star_min,
         y_star_max) = calc_delta_y_max(S_y, f_y, f_I_min[:, N - 1], f_I_max[:, N - 1])
        w_star_min = np.zeros(N, dtype=int)
        w_star_max = np.zeros(N, dtype=int)
        x_star_min = np.zeros(N, dtype=int)
        x_star_max = np.zeros(N, dtype=int)
        for i in range(N):
            w_star_min[i], x_star_min[i] = wj_xj_star_min_from_pj[p_star_min_from_y[y_star_min][i]]
            w_star_max[i], x_star_max[i] = wj_xj_star_max_from_pj[p_star_max_from_y[y_star_max][i]]
    results = dict()
    if f_y:
        results['delta_y_max_min'] = delta_y_max_min
        results['delta_y_max_max'] = delta_y_max_max
        results['y_star_min'] = y_star_min
        results['y_star_max'] = y_star_max
        results['w_x_star_min'] = (w_star_min, x_star_min)
        results['w_x_star_max'] = (w_star_max, x_star_max)
    else:
        results['w_x_star_min_from_y'] = get_w_x_star(p_star_min_from_y,
                                                      wj_xj_star_min_from_pj, S_y)
        results['w_x_star_max_from_y'] = get_w_x_star(p_star_max_from_y,
                                                      wj_xj_star_max_from_pj, S_y)  # RLine 10
    results['I_min_from_y'] = {y: f_I_min[y][N - 1] for y in S_y}
    results['I_max_from_y'] = {y: f_I_max[y][N - 1] for y in S_y}

    # Print newline on completion
    print()
    return results


def get_symmetric_number_of_iterations_dp(max_value: int, n_dim: int):
    # Initialize a DP table where n_iterations_table[i][j] represents the result
    # for max_value=i and n_dim=j
    n_iterations_table = np.zeros((max_value + 1, n_dim), dtype=np.int64)

    # Base cases
    # When n_dim = 1, the number of iterations is max_value + 1
    for i in range(max_value + 1):
        n_iterations_table[i][1 - 1] = i + 1

    # When max_value = 0, the number of iterations is always 1, regardless of n_dim
    n_iterations_table[0] = np.ones(n_dim, dtype=int)

    # Fill the DP table
    for i in range(1, max_value + 1):
        for j in range(1, n_dim):
            for k in range(i + 1):
                n_iterations_table[i][j] += n_iterations_table[i - k][j - 1]

    return n_iterations_table[max_value][n_dim - 1]


def get_next_nondescending_vector(current_vector, N: int, max_value: int):
    """
    Assuming a list of vectors where these elements of the vector are
    non-descending, this function returns the next vector in the list.

    Every element gets values from `0` to `max_value`.

    Returns the next vector and the index for bookkeeping.
    """
    # Increment the coordinates
    if current_vector[N - 1] <= max_value - 1:
        current_vector[N - 1] += 1
    else:  # current_vector[N - 1] == max_value
        # Carrying over
        # The carry is added to the position between non-max and max values
        # For example, for a vector _____adddddd, where the max values is 'd',
        # the vector is incremented to _____bbbbbbb.
        idx_to_add_carry = N - 1 - 1
        while (idx_to_add_carry >= 0
                and current_vector[idx_to_add_carry] == max_value):
            idx_to_add_carry -= 1
        if idx_to_add_carry == -1:
            return None
        next_value = current_vector[idx_to_add_carry] + 1
        for i in range(idx_to_add_carry, N):
            current_vector[i] = next_value

    return current_vector


def symmetry_based_algorithm(params: dict, f_y: callable = None):
    """
    This function implements the symmetry-based algorithm. It doesn't have its
    own algorithm in the report. See alg:naive_algorithm from the report for
    line numbers (RLine).
    """
    # Extract necessary parameters
    N = params['arch_params']['N']
    x_max = params['chip_params']['logical_max']
    w_max = params['memristor_params']['logical_max']
    # Start Algorithm
    S_y_1 = get_all_y_values_N_eq_1(w_max, x_max)  # RLine 1
    p_j_list_sorted = sorted(S_y_1)
    (f_i_min_pj,
     f_i_max_pj,
     wj_xj_star_min_from_pj,
     wj_xj_star_max_from_pj) = abstracting_function(params)  # RLine 2
    f_I_min = np.full((max(S_y_1)*N + 1), np.inf)
    f_I_max = np.zeros((max(S_y_1)*N + 1))  # RLine 3
    p_star_min_from_y = dict()
    p_star_max_from_y = dict()  # RLine 4
    S_y = set()
    vec_p_state_point = np.zeros(N, dtype=int)  # RLine 5.1
    total_iterations = get_symmetric_number_of_iterations_dp(len(S_y_1) - 1, N)
    print(f"Total iterations for symmetry-based algorithm: {total_iterations}")
    prev_percent = -1  # For progress bar
    start_time = time.time()  # For approximate remaining time calculation
    current_iteration = 0  # For progress bar
    while True:  # RLine 5.2
        # Note that S_p_j = S_y_1
        #
        # vec_p_state_point has elements in the range [0, len(S_p_j) - 1] whereas
        # vec_p has elements from S_p_j which are in the range [0, max(S_p_j)] but
        # have some holes. Therefore, we map the numbers to the S_p_j values, below.
        vec_p = [p_j_list_sorted[p_j_state_num] for p_j_state_num in vec_p_state_point]  # RLine 5.3
        y = sum(vec_p)  # RLine 6
        S_y.add(y)
        min_current_from_vec_p = sum([f_i_min_pj[p_j] for p_j in vec_p])
        max_current_from_vec_p = sum([f_i_max_pj[p_j] for p_j in vec_p])
        if min_current_from_vec_p < f_I_min[y]:
            f_I_min[y] = min_current_from_vec_p
            p_star_min_from_y[y] = vec_p
        if max_current_from_vec_p > f_I_max[y]:  # RLine 7
            f_I_max[y] = max_current_from_vec_p  # RLine 8
            p_star_max_from_y[y] = vec_p  # RLine 9

        vec_p_state_point = get_next_nondescending_vector(vec_p_state_point, N, len(S_y_1) - 1)  # RLine 5.5
        if vec_p_state_point is None:
            break  # RLine 5.7

        # Print progress bar
        prev_percent = print_progress_bar(current_iteration, total_iterations,
                                          start_time, prev_percent=prev_percent)
        current_iteration += 1
    if f_y:
        (delta_y_max_min,
         delta_y_max_max,
         y_star_min,
         y_star_max) = calc_delta_y_max(S_y, f_y, f_I_min, f_I_max)
        w_star_min = np.zeros(N, dtype=int)
        w_star_max = np.zeros(N, dtype=int)
        x_star_min = np.zeros(N, dtype=int)
        x_star_max = np.zeros(N, dtype=int)
        for i in range(N):
            w_star_min[i], x_star_min[i] = wj_xj_star_min_from_pj[p_star_min_from_y[y_star_min][i]]
            w_star_max[i], x_star_max[i] = wj_xj_star_max_from_pj[p_star_max_from_y[y_star_max][i]]
    results = dict()
    if f_y:
        results['delta_y_max_min'] = delta_y_max_min
        results['delta_y_max_max'] = delta_y_max_max
        results['y_star_min'] = y_star_min
        results['y_star_max'] = y_star_max
        results['w_x_star_min'] = (w_star_min, x_star_min)
        results['w_x_star_max'] = (w_star_max, x_star_max)
    else:
        results['w_x_star_min_from_y'] = get_w_x_star(p_star_min_from_y,
                                                      wj_xj_star_min_from_pj, S_y)
        results['w_x_star_max_from_y'] = get_w_x_star(p_star_max_from_y,
                                                      wj_xj_star_max_from_pj, S_y)  # RLine 10
    results['I_min_from_y'] = {y: f_I_min[y] for y in S_y}
    results['I_max_from_y'] = {y: f_I_max[y] for y in S_y}
    # Print newline on completion
    print()
    return results


def dynamic_programming_algorithm(params: dict, f_y: callable = None):
    """
    This function implements the dynamic programming algorithm. It is given in
    the paper (label: alg:dpba) only for finding the maximum values. Therefore,
    the line numbers (PLine) are given only for the maximum values.
    """

    def w_and_x_star_from_p_star(p_star, wj_xj_star_from_pj,):
        """
        This function returns the \vec{w} and \vec{x} vectors that correspond to
        the given p_star. It implements the equation given in the paper with
        label: eq:abstraction_unique_trace.
        """
        N = len(p_star)
        w_star = np.zeros(N, dtype=int)
        x_star = np.zeros(N, dtype=int)
        for i in range(N):
            w_star[i], x_star[i] = wj_xj_star_from_pj[p_star[i]]
        return w_star, x_star

    def trace(y: int, n: int, w_star: dict, x_star: dict, y_sub_star, n_sub_star):
        """
        This function implements the trace generation function. It is given in
        the paper with label "alg:trace_dpba".
        """
        if ((n in w_star[y]) and (n in x_star[y])):  # if w_star[y][n] and x_star[y][n] has been computed
            return
        trace(y_sub_star[y][n-1], n_sub_star[y][n-1], w_star, x_star, y_sub_star, n_sub_star)
        trace(y - y_sub_star[y][n-1], n - n_sub_star[y][n-1], w_star, x_star, y_sub_star, n_sub_star)
        w_star[y][n] = np.concatenate([
            w_star[y_sub_star[y][n-1]][n_sub_star[y][n-1]],
            w_star[y - y_sub_star[y][n-1]][n - n_sub_star[y][n-1]]
        ])
        x_star[y][n] = np.concatenate([
            x_star[y_sub_star[y][n-1]][n_sub_star[y][n-1]],
            x_star[y - y_sub_star[y][n-1]][n - n_sub_star[y][n-1]]
        ])

    # Extract necessary parameters
    N = params['arch_params']['N']
    x_max = params['chip_params']['logical_max']
    w_max = params['memristor_params']['logical_max']
    # Start Algorithm
    S_y_1 = get_all_y_values_N_eq_1(w_max, x_max)  # PLine 1
    (f_i_min_pj,
     f_i_max_pj,
     wj_xj_star_min_from_pj,
     wj_xj_star_max_from_pj) = abstracting_function(params)  # PLine 2
    f_I_min = np.full((max(S_y_1)*N + 1, N), np.inf)
    f_I_max = np.full((max(S_y_1)*N + 1, N), -np.inf)  # PLine 3
    w_star_min = {y: dict() for y in range(0, max(S_y_1)*N + 1)}
    w_star_max = {y: dict() for y in range(0, max(S_y_1)*N + 1)}  # PLine 4.1
    x_star_min = {y: dict() for y in range(0, max(S_y_1)*N + 1)}
    x_star_max = {y: dict() for y in range(0, max(S_y_1)*N + 1)}  # PLine 4.2
    y_sub_star_min = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)
    y_sub_star_max = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)  # PLine 5.1
    n_sub_star_min = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)
    n_sub_star_max = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)  # PLine 5.2
    for n in range(1, N + 1):
        f_I_min[0][n - 1] = n*f_i_min_pj[0]
        f_I_max[0][n - 1] = n*f_i_max_pj[0]  # PLine 6
        f_I_min[1][n - 1] = (n - 1)*f_i_min_pj[0] + f_i_min_pj[1]
        f_I_max[1][n - 1] = (n - 1)*f_i_max_pj[0] + f_i_max_pj[1]  # PLine 7
        # PLine 9:
        w_star_min[0][n], x_star_min[0][n] = w_and_x_star_from_p_star([0]*n, wj_xj_star_min_from_pj)
        w_star_max[0][n], x_star_max[0][n] = w_and_x_star_from_p_star([0]*n, wj_xj_star_max_from_pj)
        # PLine 10:
        w_star_min[1][n], x_star_min[1][n] = w_and_x_star_from_p_star([0]*(n - 1) + [1], wj_xj_star_min_from_pj)
        w_star_max[1][n], x_star_max[1][n] = w_and_x_star_from_p_star([0]*(n - 1) + [1], wj_xj_star_max_from_pj)
    for y in S_y_1:
        f_I_min[y][0] = f_i_min_pj[y]
        f_I_max[y][0] = f_i_max_pj[y]  # PLine 8
        # PLine 11:
        w_star_min[y][1], x_star_min[y][1] = w_and_x_star_from_p_star([y], wj_xj_star_min_from_pj)
        w_star_max[y][1], x_star_max[y][1] = w_and_x_star_from_p_star([y], wj_xj_star_max_from_pj)
    prev_percent = -1  # For progress bar
    start_time = time.time()  # For approximate remaining time calculation
    for n in range(2, N + 1):  # PLine 12
        for y in range(2, n*w_max*x_max + 1):  # PLine 13 # y_max(n) = n*w_max*x_max
            for n_sub in range(1, n):  # PLine 14
                for y_sub in range((y/2).__ceil__(), y + 1):  # PLine 15
                    if f_I_min[y][n - 1] > f_I_min[y_sub][n_sub - 1] + f_I_min[y - y_sub][n - n_sub - 1]:
                        f_I_min[y][n - 1] = f_I_min[y_sub][n_sub - 1] + f_I_min[y - y_sub][n - n_sub - 1]
                        y_sub_star_min[y][n - 1] = y_sub
                        n_sub_star_min[y][n - 1] = n_sub
                    # PLine 16:
                    if f_I_max[y][n - 1] < f_I_max[y_sub][n_sub - 1] + f_I_max[y - y_sub][n - n_sub - 1]:
                        # PLine 17:
                        f_I_max[y][n - 1] = f_I_max[y_sub][n_sub - 1] + f_I_max[y - y_sub][n - n_sub - 1]
                        y_sub_star_max[y][n - 1] = y_sub  # PLine 18.1
                        n_sub_star_max[y][n - 1] = n_sub  # PLine 18.2

            prev_percent = print_progress_bar(n**4, N**4, start_time,
                                              prev_percent=prev_percent)
    # Print newline after completion of progress bar
    print()
    # Equivalent to PLine 19 and 20:
    S_y = {y for y in range(N*x_max*w_max + 1) if f_I_max[y][N - 1] > -np.inf}
    if f_y:
        (delta_y_max_min,
         delta_y_max_max,
         y_star_min,
         y_star_max) = calc_delta_y_max(S_y, f_y, f_I_min[:, N - 1], f_I_max[:, N - 1])  # PLine 21
        trace(y_star_min, N, w_star_min, x_star_min, y_sub_star_min, n_sub_star_min)
        trace(y_star_max, N, w_star_max, x_star_max, y_sub_star_max, n_sub_star_max)  # PLine 22
    else:
        for y in S_y:
            trace(y, N, w_star_min, x_star_min, y_sub_star_min, n_sub_star_min)
            trace(y, N, w_star_max, x_star_max, y_sub_star_max, n_sub_star_max)
    results = dict()
    if f_y:
        results['delta_y_max_min'] = delta_y_max_min
        results['delta_y_max_max'] = delta_y_max_max
        results['y_star_min'] = y_star_min
        results['y_star_max'] = y_star_max
        results['w_x_star_min'] = (w_star_min[y_star_min][N], x_star_min[y_star_min][N])
        results['w_x_star_max'] = (w_star_max[y_star_max][N], x_star_max[y_star_max][N])
    else:
        w_x_star_min_from_y = dict()
        w_x_star_max_from_y = dict()
        for y in S_y:  # RLine 2
            w_x_star_min_from_y[y] = (w_star_min[y][N], x_star_min[y][N])
            w_x_star_max_from_y[y] = (w_star_max[y][N], x_star_max[y][N])
        results['w_x_star_min_from_y'] = w_x_star_min_from_y
        results['w_x_star_max_from_y'] = w_x_star_max_from_y
    results['I_min_from_y'] = {y: f_I_min[y][N - 1] for y in S_y}
    results['I_max_from_y'] = {y: f_I_max[y][N - 1] for y in S_y}
    return results


def dynamic_programming_algorithm_with_abstraction(params: dict, f_y: callable = None):
    """
    This function implements the dynamic programming algorithm with abstraction.
    It is given in the report (label: alg:dynamic_programming_algorithm) and
    paper (label: alg:dpba but without abstraction) only for finding the maximum
    values. Therefore, the line numbers (RLine and PLine, respectively) are
    given only for the maximum values.
    """

    def trace(y: int, n: int, p_star: dict, y_sub_star, n_sub_star):
        """
        This function implements the trace generation function. It is given in
        the paper with label "alg:trace_dpba".
        """
        if n in p_star[y]:  # if p_star[y][n] has been computed
            return
        trace(y_sub_star[y][n-1], n_sub_star[y][n-1], p_star, y_sub_star, n_sub_star)
        trace(y - y_sub_star[y][n-1], n - n_sub_star[y][n-1], p_star, y_sub_star, n_sub_star)
        p_star[y][n] = (p_star[y_sub_star[y][n-1]][n_sub_star[y][n-1]]
                        + p_star[y - y_sub_star[y][n-1]][n - n_sub_star[y][n-1]])

    # Extract necessary parameters
    N = params['arch_params']['N']
    x_max = params['chip_params']['logical_max']
    w_max = params['memristor_params']['logical_max']
    # Start Algorithm
    S_y_1 = get_all_y_values_N_eq_1(w_max, x_max)  # PLine 1, RLine 1
    (f_i_min_pj,
     f_i_max_pj,
     wj_xj_star_min_from_pj,
     wj_xj_star_max_from_pj) = abstracting_function(params)  # RLine 2
    f_I_min = np.full((max(S_y_1)*N + 1, N), np.inf)
    f_I_max = np.full((max(S_y_1)*N + 1, N), -np.inf)  # PLine 3, RLine 3
    p_star_min = {y: dict() for y in range(0, max(S_y_1)*N + 1)}
    p_star_max = {y: dict() for y in range(0, max(S_y_1)*N + 1)}  # RLine 4
    y_sub_star_min = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)
    y_sub_star_max = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)  # PLine 5.1
    n_sub_star_min = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)
    n_sub_star_max = np.full((max(S_y_1)*N + 1, N), -1, dtype=int)  # PLine 5.2
    for n in range(1, N + 1):
        f_I_min[0][n - 1] = n*f_i_min_pj[0]
        f_I_max[0][n - 1] = n*f_i_max_pj[0]  # PLine 6, RLine 5
        f_I_min[1][n - 1] = (n - 1)*f_i_min_pj[0] + f_i_min_pj[1]
        f_I_max[1][n - 1] = (n - 1)*f_i_max_pj[0] + f_i_max_pj[1]  # PLine 7, RLine 6
        p_star_min[0][n] = [0]*n
        p_star_max[0][n] = [0]*n  # RLine 8
        p_star_min[1][n] = [0]*(n - 1) + [1]
        p_star_max[1][n] = [0]*(n - 1) + [1]  # RLine 9
    for y in S_y_1:
        f_I_min[y][0] = f_i_min_pj[y]
        f_I_max[y][0] = f_i_max_pj[y]  # PLine 8, RLine 7
        p_star_min[y][1] = [y]
        p_star_max[y][1] = [y]  # RLine 10
    prev_percent = -1  # For progress bar
    start_time = time.time()  # For approximate remaining time calculation
    for n in range(2, N + 1):  # PLine 12, RLine 11
        for y in range(2, n*w_max*x_max + 1):  # PLine 13, RLine 12 # y_max(n) = n*w_max*x_max
            for n_sub in range(1, n):  # PLine 14, RLine 15
                for y_sub in range((y/2).__ceil__(), y + 1):  # PLine 15, RLine 16
                    if f_I_min[y][n - 1] > f_I_min[y_sub][n_sub - 1] + f_I_min[y - y_sub][n - n_sub - 1]:
                        f_I_min[y][n - 1] = f_I_min[y_sub][n_sub - 1] + f_I_min[y - y_sub][n - n_sub - 1]
                        y_sub_star_min[y][n - 1] = y_sub
                        n_sub_star_min[y][n - 1] = n_sub
                    # PLine 16, RLine 17:
                    if f_I_max[y][n - 1] < f_I_max[y_sub][n_sub - 1] + f_I_max[y - y_sub][n - n_sub - 1]:
                        # PLine 17, RLine 18:
                        f_I_max[y][n - 1] = f_I_max[y_sub][n_sub - 1] + f_I_max[y - y_sub][n - n_sub - 1]
                        y_sub_star_max[y][n - 1] = y_sub  # PLine 18.1
                        n_sub_star_max[y][n - 1] = n_sub  # PLine 18.2

            prev_percent = print_progress_bar(n**4, N**4, start_time,
                                              prev_percent=prev_percent)
    # Print newline after completion of progress bar
    print()
    # Equivalent to PLine 19 and 20:
    S_y = {y for y in range(N*x_max*w_max + 1) if f_I_max[y][N - 1] > -np.inf}
    if f_y:
        (delta_y_max_min,
         delta_y_max_max,
         y_star_min,
         y_star_max) = calc_delta_y_max(S_y, f_y, f_I_min[:, N - 1], f_I_max[:, N - 1])  # PLine 21
        trace(y_star_min, N, p_star_min, y_sub_star_min, n_sub_star_min)
        trace(y_star_max, N, p_star_max, y_sub_star_max, n_sub_star_max)  # PLine 22
        w_star_min = np.zeros(N, dtype=int)
        w_star_max = np.zeros(N, dtype=int)
        x_star_min = np.zeros(N, dtype=int)
        x_star_max = np.zeros(N, dtype=int)
        for i in range(N):
            w_star_min[i], x_star_min[i] = wj_xj_star_min_from_pj[p_star_min[y_star_min][N][i]]
            w_star_max[i], x_star_max[i] = wj_xj_star_max_from_pj[p_star_max[y_star_max][N][i]]
    else:
        for y in S_y:
            trace(y, N, p_star_min, y_sub_star_min, n_sub_star_min)
            trace(y, N, p_star_max, y_sub_star_max, n_sub_star_max)
    results = dict()
    if f_y:
        results['delta_y_max_min'] = delta_y_max_min
        results['delta_y_max_max'] = delta_y_max_max
        results['y_star_min'] = y_star_min
        results['y_star_max'] = y_star_max
        results['w_x_star_min'] = (w_star_min, x_star_min)
        results['w_x_star_max'] = (w_star_max, x_star_max)
    else:
        results['w_x_star_min_from_y'] = get_w_x_star({y: p_star_min[y][N] for y in S_y},
                                                      wj_xj_star_min_from_pj, S_y)
        results['w_x_star_max_from_y'] = get_w_x_star({y: p_star_max[y][N] for y in S_y},
                                                      wj_xj_star_max_from_pj, S_y)  # RLine 20
    results['I_min_from_y'] = {y: f_I_min[y][N - 1] for y in S_y}
    results['I_max_from_y'] = {y: f_I_max[y][N - 1] for y in S_y}
    return results


def verify_column(results: dict):
    log_file_name = 'memristor_column_verification_results'
    verify_logger = setup_logger(log_file_name)
    L_y = sorted(results['I_min_from_y'].keys())
    for i_y, y in enumerate(L_y):
        if y == 0:
            continue
        y_pre = L_y[i_y - 1]
        if results['I_max_from_y'][y_pre] > results['I_min_from_y'][y]:
            verify_logger.info(
                f"Verification failed for y = {y_pre} and y = {y} since:\n"
                f"I_max({y_pre}) = {results['I_max_from_y'][y_pre]} > {results['I_min_from_y'][y]} = I_min({y})\n"
                f"w for I_max({y_pre}): {results['w_x_star_max_from_y'][y_pre][0]}\n"
                f"x for I_max({y_pre}): {results['w_x_star_max_from_y'][y_pre][1]}\n"
                f"w I_min({y}): {results['w_x_star_min_from_y'][y][0]}\n"
                f"x I_min({y}): {results['w_x_star_min_from_y'][y][1]}"
            )
            print(
                f"!Warning: The memristor column does not work correctly. "
                f"Please check the {log_file_name}.log file for details."
            )
            return False
    verify_logger.info("Verification of memristor column is successful!")
    return True


def calc_delta_y_max(S_y, f_y: callable, f_I_min_from_y, f_I_max_from_y):
    delta_y_max_min = -1
    delta_y_max_max = -1
    for y in S_y:
        delta_y_max_min_cadidate = abs(y - f_y(f_I_min_from_y[y]))
        delta_y_max_max_cadidate = abs(y - f_y(f_I_max_from_y[y]))
        if delta_y_max_min_cadidate > delta_y_max_min:
            delta_y_max_min = delta_y_max_min_cadidate
            y_star_min = y
        if delta_y_max_max_cadidate > delta_y_max_max:
            delta_y_max_max = delta_y_max_max_cadidate
            y_star_max = y
    return delta_y_max_min, delta_y_max_max, y_star_min, y_star_max
