# Formal Verification of Error Bounds for Resistive-Switching-based Multilevel Matrix-Vector Multipliers

This project implements methods in [1] to verify the correctness of analog MAC units based on memristor crossbar arrays. The methods are implemented in Python and are compared for their execution times. Additionally, the results of the methods are compared to verify their correctness. If used, please cite the following paper:

[1] K. Ç. Coşkun, C. K. Jha, M. Hassan, and R. Drechsler, “Formal Verification of Error Bounds for Resistive-Switching-based Multilevel Matrix-Vector Multipliers,” in 2025 26th International Symposium on Quality Electronic Design (ISQED) (Accepted), 2025.

## Requirements

See the `requirements.txt` file for the required packages.

## Installation

Clone the repository and install the required packages and this package:

```bash
git clone https://github.com/KCaglarCoskun/multi-level-mvm-verification.git
pip install -r requirements.txt
pip install .
```
Or alternatively install in a virtual environment.

The package was tested with Python 3.12, numpy 2.2.2, and PyYAML 6.0.2. The example in the folder case_study_MJW_2023 requires matplotlib and was tested with version 3.10.0.

## Usage

To run the analysis, create a folder with the following files:

- `[your_configuration_file].yaml`: Contains the parameters for the calculations.
- `[your_main_file].py`: Contains the main script that loads the parameters and executes the methods.

and run the main script.

If in doubt, use the examples in the `examples` folder.

### Parameters File

The `example_parameters.yaml` file contains example values that serve as a template. Modify these values according to the memristor crossbar array you want to analyze.

Make sure to preserve the input format for the parameters.

### Main Script File

Load the parameters with the `utils.config_loader.load_config` function:

```python
from multilevelmvmverification.utils.config_loader import load_config

params = load_config('[your_configuration_file].yaml')
```

The software can be used either (1) to find the maximum error that can occur for the given MVM (based on its params and an output function $f_y$), or (2) to verify that there exist an $f_y$ such that the MVM will work without any errors.

#### (1) First Use Case
For the first use case, call a `methods.<method_name>` function with the params and a current to output function, $f_y$. The variable `results` will have many results. For a human readable form, run this variable through the `trace_logger` function.

```python
from multilevelmvmverification.methods import <method_name>
from multilevelmvmverification.utils.logger import trace_logger

results = <method_name>(params, current_to_output_function)
trace_logger(results)
```

For the method name always use dynamic_programming_algorithm. Other two possibilities exist solely for comparison purposes: brute_force_algorithm, symmetry_based_algorithm.

#### (2) Second Use Case
For the second use case, call a `methods.<method_name>` function with the params. The variable `results` will have many results. For a human readable form, run this variable through the `verify_column` function.

```python
from multilevelmvmverification.methods import <method_name>
from multilevelmvmverification.methods import verify_column

results = <method_name>(params)
verify_column(results)
```

### Output Logs

The software prints some general information to the console, but writes more detailed information into a folder called `output`. Some of these files are:

- `config_validation.log`: Contains information about the validity of the configuration file.
- `performance.log`: Contains information about each method's execution time.
- `method_equivalence_results.log`: Logs the outcome of the results comparison among different methods.
- `memristor_column_verification_results.log`: Logs the outcome of the verification for a memristor column.
- `currents_vs_y.csv`: Contains the minimum and maximum currents for each possible logical output value.

To create the `currents_vs_y.csv` file, do:
    
```python
from multilevelmvmverification.utils.logger import export_currents_to_csv

export_currents_to_csv(results)
```

## Structure

The project is organized as follows:

- `example_parameters.yaml`: Example configuration file to set parameters for calculations.
- `methods.py`: Contains different methods for performance analysis.
- `utils/`: Houses utility scripts including the configuration loader, configuration verifier, logger, timer, and result verifier.
- `tests/`: Contains unit tests for some helper functions.
- `examples/`: Contains an example case study and scripts to run the software.
- `timing_analysis_and_validation/`: Contains scripts to compare the performance of the methods and validate the results.
- `requirements.txt`: Contains the required packages for the software.
- `output/`: Contains the output logs (not in source control).