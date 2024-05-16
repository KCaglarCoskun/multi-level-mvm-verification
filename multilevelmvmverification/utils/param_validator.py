from multilevelmvmverification.utils.logger import setup_logger

config_logger = setup_logger('config_validation')
nonnegative_number_parameters = {'physical_0_nom', 'physical_max_nom', 'relative_error'}
positive_integer_parameters = {'N', 'n_cols', 'logical_max'}
list_of_nonnegative_number_parameters = {'physical_noms'}
multiple_choice_parameters = {'distribution': {'linear', 'log'}}


def validate_value_types(params: dict, parent_key: str = ''):
    for key, value in params.items():
        key_with_parent = f"{parent_key} -> {key}" if parent_key != '' else key
        if isinstance(value, dict):
            validate_value_types(value, parent_key=key)
        else:
            if key in nonnegative_number_parameters:
                assert isinstance(value, (int, float)), (
                    f"Parameter '{key_with_parent}' should be a number. Found: {type(value).__name__}"
                )
                assert value >= 0, f"Parameter '{key_with_parent}' should be non-negative. Found: {value}"
            elif key in positive_integer_parameters:
                assert isinstance(value, int), (
                    f"'{key_with_parent}' should be an integer. Found: {type(value).__name__}"
                )
                assert value > 0, f"Parameter '{key_with_parent}' should be positive. Found: {value}"
            elif key in list_of_nonnegative_number_parameters:
                assert isinstance(value, list), (
                    f"Parameter '{key_with_parent}' should be a list. Found: {type(value).__name__}"
                )
                for i, item in enumerate(value):
                    assert isinstance(item, (int, float)), (
                        f"Parameter '{key_with_parent}' value at index {i} should be a number. "
                        f"Found: {type(item).__name__}"
                    )
                    assert item >= 0, (
                        f"Parameter '{key_with_parent}' value at index {i} should be non-negative. Found: {item}"
                    )
            elif key in multiple_choice_parameters:
                assert value in multiple_choice_parameters[key], (
                    f"Parameter '{key_with_parent}' should be one of {multiple_choice_parameters[key]}. "
                    f"Found: {value}"
                )


def validate_params(params):
    try:
        validate_value_types(params)
        assert 'arch_params' in params, "Missing 'arch_params' section in configuration."
        validate_arch_params(params['arch_params'])

        for technology_key in ('chip_params', 'memristor_params'):
            assert technology_key in params, f"Missing '{technology_key}' section in configuration."
            validate_technology_params(params[technology_key], technology_key)

        return True
    except AssertionError as error:
        config_logger.error(f"Config Validation Error: {error}")
        return False


def validate_arch_params(arch_params):
    required_keys = ['N', 'n_cols']
    for key in required_keys:
        assert key in arch_params, f"Missing '{key}' in architecture parameters."
    assert arch_params['n_cols'] == 1, (
        f"Values other than 1 for 'n_cols' in architecture parameters is not yet supported. "
        f"Found: {arch_params[key]}"
    )


def validate_technology_params(technology_params, technology_key):
    if (('physical_noms' not in technology_params)
            and ('physical_maxs' not in technology_params)
            and ('physical_mins' not in technology_params)):
        # if these keys are not provided, logical_max, physical_0_nom,
        # physical_max_nom, and distribution should be provided
        required_for_all = ('logical_max', 'physical_max_nom', 'distribution')
        for key in required_for_all:
            assert key in technology_params, (
                f"Parameter '{technology_key} -> {key}' should be provided "
                f"if '{technology_key} -> physical_noms' is not provided."
            )
        # physical_0_nom can always be skipped for chip_params
        if technology_key == 'memristor_params':
            assert 'physical_0_nom' in technology_params, (
                f"Parameter '{technology_key} -> physical_0_nom' should be provided "
                f"if '{technology_key} -> physical_noms' is not provided."
            )

    if (('physical_maxs' not in technology_params)
            and ('physical_mins' not in technology_params)):
        assert 'relative_error' in technology_params, (
            f"Parameter '{technology_key} -> relative_error' missing in chip parameters."
        )
        assert 0 <= technology_params['relative_error'] <= 1, (
            f"Parameter '{technology_key} -> relative_error' should be between 0 and 1. "
            f"Found: {technology_params['v_relative_error']}"
        )
