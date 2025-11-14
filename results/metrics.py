
import math

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from collections import defaultdict
from evaluation.evaluation_for_autonomous_driving import simulate_policy

from config.parameters import *



def performance_metrics_over_uncertainties(
    N_sim,
    policy,
    should_use_rescale_action_wrapper,
    bicycle_model_parameters,
    road_elements_list,
    numerical_integration_parameters,
    termination_parameters,
    initial_state_bounds,
    observation_parameters,
    num_sim=50,
):
    # Initialise a list for all the performance metrics
    pm_per_sim_list = []
    path_per_sim_list = []

    # Specify the range of uncertanties
    mass_multipliers = np.arange(start=0.95, stop=1.051, step=0.05)
    sim_range = range (0,num_sim)

    # Iterate over the perturbations of model parameters
    with tqdm(total=len(mass_multipliers) * len(sim_range)) as pbar:
        for mass_multiplier in mass_multipliers:
            # Create an environment with the adjusted mass parameter
            # > Take a copy of the existing model parameters
            bicycle_model_parameters_adjusted = bicycle_model_parameters.copy()
            # > Adjust the mass parameter
            bicycle_model_parameters_adjusted["m"] = mass_multiplier * bicycle_model_parameters["m"]
            # > Make the environment
            env_adjusted = gym.make(
                "ai4rgym/autonomous_driving_env",
                render_mode=None,
                bicycle_model_parameters=bicycle_model_parameters_adjusted,
                road_elements_list=road_elements_list,
                numerical_integration_parameters=numerical_integration_parameters,
                termination_parameters=termination_parameters,
                initial_state_bounds=initial_state_bounds,
                observation_parameters=observation_parameters,
            )
            # > Add the rescale action wrapper (if requested)
            if (should_use_rescale_action_wrapper):
                env_adjusted = gym.wrappers.RescaleAction(env_adjusted, min_action=-1, max_action=1)

            # Initialise a seed for the random number generators
            sim_seed = seed

            # Iterate over 100 simulation
            for i_sim in sim_range:
                # Reset the policy (if it has a reset function)
                if hasattr(policy, 'reset'):
                    policy.reset()
                # Increment the seed
                sim_seed += 1
                # Call the simulation function
                sim_time_series_dict = simulate_policy(env_adjusted, N_sim, policy, seed=sim_seed)
                # Compute the performance metrics for this simulation
                pm_dict = compute_performance_metrics_from_time_series(sim_time_series_dict)
                # Compute the path for this simulation
                path_dict = compute_path_from_time_series(sim_time_series_dict)
                # Append to the list of all performance metrics
                pm_per_sim_list.append( pm_dict )
                path_per_sim_list.append( path_dict )

                # Update the progress bar
                pbar.update(1)

    # Create arrays padded with NaN
    max_len = max(len(p["x"]) for p in path_per_sim_list)
    all_x = np.full((len(path_per_sim_list), max_len), np.nan)
    all_y = np.full((len(path_per_sim_list), max_len), np.nan)

    for i, p in enumerate(path_per_sim_list):
        all_x[i, :len(p["x"])] = p["x"]
        all_y[i, :len(p["y"])] = p["y"]

    mean_x = np.nanmean(all_x, axis=0).tolist()
    mean_y = np.nanmean(all_y, axis=0).tolist()
    std_x = np.nanstd(all_x, axis=0).tolist()
    std_y = np.nanstd(all_y, axis=0).tolist()

    # Return the list of performance metrics per simulation
    return pm_per_sim_list, { "mean_x": mean_x, "mean_y": mean_y, "std_x": std_x, "std_y": std_y }



def compute_path_from_time_series(sim_time_series_dict):
    xs = [x for x in sim_time_series_dict["px"].tolist() if not (isinstance(x, float) and math.isnan(x))]
    ys = [y for y in sim_time_series_dict["py"].tolist() if not (isinstance(y, float) and math.isnan(y))]
    return {
        "x": np.array(xs, dtype=float),
        "y": np.array(ys, dtype=float),
    }



def compute_performance_metrics_from_time_series(sim_time_series_dict):
    speed_time_series = np.abs(sim_time_series_dict["vx"])
    #rec_speed_time_series = sim_time_series_dict["recommended_speed"]
    rec_speed_time_series = 60 / 3.6
    dist_to_line_time_series = sim_time_series_dict["distance_to_closest_point"]
    heading_angle_relative_to_line_time_series = sim_time_series_dict["heading_angle_relative_to_line"]
    delta_time_series = sim_time_series_dict["delta"]
    time_time_series = sim_time_series_dict["time_in_seconds"]

    delta_t = time_time_series[1:] - time_time_series[:-1]

    # Requirement 1: Do not exceed the speed limit or the recommended speed (whichever is lower).
    r1 = -np.nansum(np.minimum(0, rec_speed_time_series - speed_time_series))

    # Requirement 2: Generally drive at a speed that is within a few km/h of the speed limit or the recommended speed (whichever is lower).
    r2 = np.nansum((speed_time_series - rec_speed_time_series)**2)

    # Requirement 3: Smooth lane-keeping
    alpha = 1.0
    beta = 1.0

    r31 = np.nansum(alpha * dist_to_line_time_series[1:]**2 + beta * heading_angle_relative_to_line_time_series[1:]**2)
    r32 = np.nansum(((delta_time_series[1:] - delta_time_series[:-1]) / delta_t)**2)

    # Requirement 4: Smooth acceleration and deceleration.
    r4 = np.nansum(np.abs((speed_time_series[2:] - 2 * speed_time_series[1:-1] + speed_time_series[:-2])/(delta_t[1:]**2)))

    # Return the results
    return {
        "r1"   :  r1,
        "r2"   :  r2,
        "r3.1" :  r31,
        "r3.2" :  r32,
        "r4"   :  r4,
    }



def identify_pareto_front(results):
    # Group results by controller parameters (excluding road_elements_name)
    grouped = defaultdict(list)

    for result in results:
        grouped[result["model_name"]].extend(result["performance_metrics"])

    # Now compute averages for each group
    pm_per_policy_list = {}
    for key, dicts in grouped.items():
        # convert list of dicts to numpy array
        arr = np.array([[d["r1"], d["r2"], d["r3.1"], d["r3.2"], d["r4"]] for d in dicts])
        mean_vals = arr.mean(axis=0)
        pm_per_policy_list[key] = {
            "r1"    : mean_vals[0],
            "r2"    : mean_vals[1],
            "r3.1"  : mean_vals[2],
            "r3.2"  : mean_vals[3],
            "r4"    : mean_vals[4],
        }

    # Specify a dictionary of which direction is better for each metric
    # These should be the same keys are used in the performance metric dictionaries
    pm_desired_direction = {
        "r1"    : "smaller",
        "r2"    : "smaller",
        "r3.1"  : "smaller",
        "r3.2"  : "smaller",
        "r4"    : "smaller",
    }

    # Extract the performance metric keys from the dictionary
    pm_keys = pm_desired_direction.keys()

    # Initialise a list for whether the policy is "dominiated" or on the "front"
    policy_is_pareto_list = []

    # Initialise lists for which policies something is dominated-by or dominates
    policy_is_dominated_by_list = [[] for _ in range(len(pm_per_policy_list))]
    policy_dominates_list       = [[] for _ in range(len(pm_per_policy_list))]

    # Iterate over the policies
    for i, i_pm_dict_key in enumerate(pm_per_policy_list, start=0):
        # Default it to be on the front
        i_policy_is_pareto = "front"

        i_pm_dict = pm_per_policy_list[i_pm_dict_key]

        # Iterate over the "other" polcies to check if they dominate it
        for j, j_pm_dict_key in enumerate(pm_per_policy_list, start=0):
            # Skip if i equals j because we don't need to compare a policy to itself
            if (i == j):
                continue

            j_pm_dict = pm_per_policy_list[j_pm_dict_key]

            # Default to "j dominates i"
            j_policy_dominates_i = True
            # Iterate over the keys
            for key in pm_keys:
                # Check if "j dominates i" is not true
                if (pm_desired_direction[key] == "bigger"):
                    if (i_pm_dict[key] > j_pm_dict[key]):
                        j_policy_dominates_i = False
                        break
                elif (pm_desired_direction[key] == "smaller"):
                    if (i_pm_dict[key] < j_pm_dict[key]):
                        j_policy_dominates_i = False
                        break

            # Update the "i_policy_is_pareto"
            if (j_policy_dominates_i):
                i_policy_is_pareto = "dominated"
                policy_is_dominated_by_list[i].append(j)
                policy_dominates_list[j].append(i)

        # Append the result for policy i
        policy_is_pareto_list.append(i_policy_is_pareto)

    # Display the results
    print("Policy is pareto:")
    print(policy_is_pareto_list)
    print("")
    print("Policy is dominated by:")
    print(policy_is_dominated_by_list)
    print("")
    print("Policy dominates:")
    print(policy_dominates_list)
