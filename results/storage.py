import json
import os
import pandas as pd



PATH = "./results/data/"



def merge_files(file_name_1, file_name_2, method_name):
    data1, models1 = load_performance_metrics(file_name=file_name_1, method_name=method_name)
    data2, models2 = load_performance_metrics(file_name=file_name_2, method_name=method_name)

    data = data1 + data2
    models = models1 + models2

    return store_performance_metrics(data, models, method_name=method_name)




def load_performance_metrics(file_name, method_name, input_path=None):
    if input_path is None:
        input_path = os.path.join(PATH, method_name)

    with open(os.path.join(input_path, f"{method_name}_data_{file_name}.json"), "r") as f:
        data = json.load(f)

    return data["pm_per_sim_lists"], data["models"]



def store_performance_metrics(pm_per_sim_lists, models, method_name, output_path=None):
    data = {
        "pm_per_sim_lists": pm_per_sim_lists,
        "models": models
    }

    if output_path is None:
        output_path = os.path.join(PATH, method_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    name = f"{pd.Timestamp.now().strftime("%m_%d_%H_%M")}"
    with open(os.path.join(output_path, f"{method_name}_data_{name}.json"), "w") as f:
        json.dump(data, f, default=float)

    return name
