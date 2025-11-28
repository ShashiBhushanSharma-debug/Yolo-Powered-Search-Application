import yaml

# We can ask the user to pass the config path of the model
def load_config(config_path ="configs\default.yaml"):
    with open(config_path, mode='r') as f:
        config = yaml.safe_load(f)
    return config

# If I want to make changes to the config file and write a new config then i can use the save_config funtion to save the new config

def save_config(config, config_path = "configs\defaults.yaml"):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)