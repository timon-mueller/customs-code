import os
import yaml

def load_config(config_path="config/config.yaml"):
    """
    Loads a YAML configuration file and resolves paths relative to the project root.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    

    abs_config_path = os.path.join(project_root, config_path)
    

    with open(abs_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        

    if 'dataset' in config and 'example_data_dir' in config['dataset']:
        rel_path = config['dataset']['example_data_dir']
        config['dataset']['example_data_dir'] = os.path.join(project_root, rel_path)
        
    return config