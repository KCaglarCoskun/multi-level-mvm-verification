import numpy as np
import matplotlib.pyplot as plt

from multilevelmvmverification.methods import dynamic_programming_algorithm, verify_column
from multilevelmvmverification.utils.config_loader import load_config
from multilevelmvmverification.utils.timer import time_method
from multilevelmvmverification.utils.logger import trace_logger, export_currents_to_csv


def print_f_y_vs_real(data_x_vec, data_y_vec, f_y, filename_postfix=''):
    # Plotting for checking the regression results
    regression_x_vec = np.array(sorted(data_x_vec))
    regression_y_vec = [f_y(x) for x in regression_x_vec]
    # Reduce height of the figure in points
    plt.rcParams['figure.figsize'] = [6.4, 3]
    # Set font to serif
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.plot(data_y_vec, 1e6*data_x_vec, 'b.') # 1e6 is for converting A to µA
    plt.plot(regression_y_vec, 1e6*regression_x_vec, 'r')
    plt.xlabel('Logical Output Value')
    plt.ylabel('Output Current (µA)')
    # Reduce white space around the figure
    plt.tight_layout()
    # Crop white space around the figure
    plt.savefig(f'output/memristor_column_regression{filename_postfix}.pdf', bbox_inches='tight')


def print_f_y_coeffs_to_console(coeffs, degree, console_prefix=''):
    coeff_string = f'{console_prefix}y = '
    for i, coeff in enumerate(coeffs):
        coeff_string += f'{coeff}x^{degree-i}'
        if i < degree:
            coeff_string += ' + '
    print(coeff_string)

def get_current_to_output_function_pol(results: dict, degree=2):
    # Prepare data for regression
    data_x_vec = np.array(list(results['I_min_from_y'].values()) + list(results['I_max_from_y'].values()))
    data_y_vec = np.array(list(results['I_min_from_y'].keys()) + list(results['I_max_from_y'].keys()))
    
    # Calculating coefficients using NumPy's least squares method
    coeffs = np.polyfit(data_x_vec, data_y_vec, degree)

    f_y = lambda current: sum([coeff*current**(degree-i) for i, coeff in enumerate(coeffs)])

    print_f_y_coeffs_to_console(coeffs, degree)
    
    return f_y


def get_current_to_output_function_pol_from_midpoints(results: dict, degree=2, min_weight=0.5, console_prefix='', map_zero_to_zero=True):
    # Create degree number of data points
    last_y = list(results['I_min_from_y'].keys())[-1]
    regression_y_vec = np.linspace(0, last_y, degree + 1, dtype=int)
    regression_x_vec_list = [min_weight*results['I_min_from_y'][y] + (1-min_weight)*results['I_max_from_y'][y] for y in regression_y_vec]
    if map_zero_to_zero:
        regression_x_vec_list[0] = 0
    regression_x_vec = np.array(regression_x_vec_list)
    
    # Calculating coefficients using NumPy's least squares method
    coeffs = np.polyfit(regression_x_vec, regression_y_vec, degree)

    f_y = lambda current: sum([coeff*current**(degree-i) for i, coeff in enumerate(coeffs)])

    print_f_y_coeffs_to_console(coeffs, degree, console_prefix=console_prefix)
    
    return f_y
    

def main():
    params = load_config('parameters.yaml')

    if params is None:
        return
    case_studies = [(0, None), (0, 3.45e-6), (0.1, None), (0.3, None)]
    for relative_error, new_f_G_3 in case_studies:
        for technology_key in ('chip_params', 'memristor_params'):
            if new_f_G_3:
                physical_noms = params['memristor_params']['physical_noms']
                physical_noms[3] = new_f_G_3
                params['memristor_params']['physical_noms'] = physical_noms
            params[technology_key]['physical_maxs'] = [
                physical_nom * (1 + relative_error)
                for physical_nom in params[technology_key]['physical_noms']
            ]
            params[technology_key]['physical_mins'] = [
                physical_nom * (1 - relative_error)
                for physical_nom in params[technology_key]['physical_noms']
            ]

        if (relative_error == 0) and (new_f_G_3 is None):
            degree = 2
            min_weight = 0.6
        elif (relative_error == 0) and (new_f_G_3):
            degree = 2
            min_weight = 0.58
        elif relative_error == 0.1:
            degree = 2
            min_weight = 0.6
        elif relative_error == 0.3:
            degree = 2
            min_weight = 0.63
        results = {'dynamic_programming_algorithm': time_method(dynamic_programming_algorithm, params)}
        map_zero_to_zero = False if new_f_G_3 else True
        f_y = get_current_to_output_function_pol_from_midpoints(
                                            results['dynamic_programming_algorithm'],
                                            degree=degree,
                                            min_weight=min_weight,
                                            console_prefix=f'Rel. Err: {relative_error}, f_G_3: {new_f_G_3} --> ',
                                            map_zero_to_zero=map_zero_to_zero
                                            )
        # Overwrite current to output function
        if (relative_error == 0) and (new_f_G_3 is None):
            f_y = lambda current: min(-1.34e9*current**2 + 8.2e5*current, 96)
        elif (relative_error == 0) and (new_f_G_3):
            f_y2 = f_y
            f_y = lambda current: max(min(f_y2(current), 96),0)
        elif relative_error == 0.1:
            f_y2 = f_y
            f_y = lambda current: min(f_y2(current), 96)
            # f_y = lambda current: min(156443943.92380145*current**2 + 879705.7179344323*current, 96)
        elif relative_error == 0.3:
            f_y = lambda current: min(-1.21e9*current**2 + 8.3e5*current, 96)
        results = {'dynamic_programming_algorithm': time_method(dynamic_programming_algorithm, params, f_y=f_y)}

        # Save results to file
        trace_logger(results['dynamic_programming_algorithm'], filename_postfix = f'_rel_err_{relative_error}_f_G_3_{new_f_G_3}')
        # export_currents_to_csv(results['dynamic_programming_algorithm'])
        data_x_vec = np.array(list(results['dynamic_programming_algorithm']['I_min_from_y'].values()) + list(results['dynamic_programming_algorithm']['I_max_from_y'].values()))
        data_y_vec = np.array(list(results['dynamic_programming_algorithm']['I_min_from_y'].keys()) + list(results['dynamic_programming_algorithm']['I_max_from_y'].keys()))
        print_f_y_vs_real(data_x_vec, data_y_vec, f_y, filename_postfix = f'_rel_err_{relative_error}_f_G_3_{new_f_G_3}')

if __name__ == "__main__":
    main()
