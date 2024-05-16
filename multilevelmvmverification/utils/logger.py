import logging
import os
import csv


def setup_logger(name, level=logging.INFO):
    filename_with_ext = name + '.log'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    log_path = os.path.join(output_dir, filename_with_ext)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(log_path, mode='w')  # mode='w' to overwrite the file
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def export_currents_to_csv(results):
    filename = 'currents_vs_y.csv'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['sep=,'])  # Excel needs this to open the file correctly
        csvwriter.writerow(['y', 'I_min', 'I_max'])

        for row in zip(results['I_min_from_y'].keys(),
                       results['I_min_from_y'].values(),
                       results['I_max_from_y'].values()):
            csvwriter.writerow(row)


def trace_logger(results: dict, filename_postfix='') -> None:
    log_file_name = f'memristor_column_max_error_results{filename_postfix}'
    verify_logger = setup_logger(log_file_name)

    necessary_keys = {'delta_y_max_min', 'delta_y_max_max', 'y_star_min', 'y_star_max',
                      'I_min_from_y', 'I_max_from_y', 'w_x_star_min', 'w_x_star_max'}
    missing_keys = necessary_keys - set(results.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in the dictionary provided to trace_logger: {missing_keys}\n"
                         "The dictionary should be returned from a verification algorithm that was\n"
                         "called with the optional argument f_y.")

    delta_y_max_min = results['delta_y_max_min']
    delta_y_max_max = results['delta_y_max_max']
    if delta_y_max_min > delta_y_max_max:
        extremum = 'min'
        # extremum_opposite = 'max'
    if delta_y_max_max > delta_y_max_min:
        extremum = 'max'
        # extremum_opposite = 'min'
    y_star = results[f'y_star_{extremum}']
    delta_y_max = results[f'delta_y_max_{extremum}']
    error_polarity = 1 if extremum == 'max' else -1
    y_with_error = y_star + error_polarity * delta_y_max

    if delta_y_max > 0:
        extremum_fullname = f'{extremum}imum'
        # extremum_opposite_fullname = f'{extremum_opposite}imum'
        current_key = f'I_{extremum}_from_y'
        # current_key_opposite = f'I_{extremum_opposite}_from_y'
        w_x_key = f'w_x_star_{extremum}'
        # w_x_key_opposite = f'w_x_star_{extremum_opposite}'
        verify_logger.info(
            f"Maximum error of the column happens for y = {y_star}, when current is {extremum_fullname}.\n"
            f"Details:\n"
            f"The {extremum_fullname} current for y = {y_star} is {results[current_key][y_star]}\n"
            f"The corresponding output results is {y_with_error} and error is {delta_y_max}\n"
            f"w for f_I_{extremum}({y_star}): {results[w_x_key][0]}\n"
            f"x for f_I_{extremum}({y_star}): {results[w_x_key][1]}\n"
            # I can't add these details when f_y is provided since I don't trace for every y
            # f"Further Details:\n"
            # f"The {extremum_opposite_fullname} current for y = {y_star} is "
            # f"{results[current_key_opposite][y_star]}\n"
            # f"w for f_I_{extremum_opposite}({y_star}): {results[w_x_key_opposite][0]}\n"
            # f"x for f_I_{extremum_opposite}({y_star}): {results[w_x_key_opposite][1]}"
        )
        print(
            f"!Warning: The memristor produces errors. "
            f"Please check the {log_file_name}.log file for details."
        )
    else:
        verify_logger.info("Memristor column works correct!")
