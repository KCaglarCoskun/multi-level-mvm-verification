import yaml
import numpy as np

from multilevelmvmverification.utils.param_validator import validate_params


def convert_resistances_to_conductances(config):
    """
    Converts the resistances in the config to conductances.
    """
    if 'physical_noms' in config['memristor_params']:
        config['memristor_params']['physical_noms'] = [1/physical_nom
                                                       for physical_nom
                                                       in config['memristor_params']['physical_noms']]
    if 'physical_maxs' in config['memristor_params']:
        config['memristor_params']['physical_maxs'] = [1/physical_max
                                                       for physical_max
                                                       in config['memristor_params']['physical_maxs']]
    if 'physical_mins' in config['memristor_params']:
        config['memristor_params']['physical_mins'] = [1/physical_min
                                                       for physical_min
                                                       in config['memristor_params']['physical_mins']]
    if 'physical_0_nom' in config['memristor_params']:
        config['memristor_params']['physical_0_nom'] = 1 / config['memristor_params']['physical_0_nom']
    if 'physical_max_nom' in config['memristor_params']:
        config['memristor_params']['physical_max_nom'] = 1 / config['memristor_params']['physical_max_nom']


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if not validate_params(config):
        print("Invalid configuration. Please check the config_validation.log file for details.")
        return

    convert_resistances_to_conductances(config)

    # Extract some useful parameters from the user provided parameters
    for technology_key in ('chip_params', 'memristor_params'):
        # We use the physical_noms list if provided
        if 'physical_noms' in config[technology_key]:
            config[technology_key]['logical_max'] = len(config[technology_key]['physical_noms']) - 1
        elif 'physical_maxs' in config[technology_key]:
            config[technology_key]['logical_max'] = len(config[technology_key]['physical_maxs']) - 1
        elif 'physical_mins' in config[technology_key]:
            config[technology_key]['logical_max'] = len(config[technology_key]['physical_mins']) - 1
        else:
            # Otherwise we use the logical_max, physical_0_nom, physical_max_nom, and distribution
            if 'physical_0_nom' not in config['chip_params']:
                config['chip_params']['physical_0_nom'] = config['chip_params']['physical_max_nom'] * (
                    config['memristor_params']['physical_0_nom'] / config['memristor_params']['physical_max_nom']
                )
            if config[technology_key]['distribution'] == 'linear':
                config[technology_key]['physical_noms'] = np.linspace(
                    config[technology_key]['physical_0_nom'],
                    config[technology_key]['physical_max_nom'],
                    config[technology_key]['logical_max'] + 1
                )
            elif config[technology_key]['distribution'] == 'log':
                config[technology_key]['physical_noms'] = np.geomspace(
                    config[technology_key]['physical_0_nom'],
                    config[technology_key]['physical_max_nom'],
                    config[technology_key]['logical_max'] + 1
                )

        if 'physical_maxs' not in config[technology_key]:
            config[technology_key]['physical_maxs'] = [
                physical_nom * (1 + config[technology_key]['relative_error'])
                for physical_nom in config[technology_key]['physical_noms']
            ]
        if 'physical_mins' not in config[technology_key]:
            config[technology_key]['physical_mins'] = [
                physical_nom * (1 - config[technology_key]['relative_error'])
                for physical_nom in config[technology_key]['physical_noms']
            ]

    return config
