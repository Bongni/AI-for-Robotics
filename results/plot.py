import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from matplotlib.collections import PolyCollection
from ai4rgym.envs.road import Road

from config.roads import *
from methods.rewards import r_lane_simple, r_speed_max, r_speed_simple, r_speed_smooth



PATH = "./results/images/"



LEGEND_TITLES_PM = {
    "rl": "Policies",
    "pid": "($P_{lk}$, $I_{lk}$, $D_{lk}$), ($P_{cc}$, $I_{cc}$, $D_{cc}$)",
    "mpc": "Policies"
}



LEGEND_MODELS_PM = {
    # RL
    "PPO simple": "Simple (PPO)",
    "PPO simple lookahead": "Simple lookahead (PPO)",
    "PPO simple sparse": "Simple sparse (PPO)",
    "PPO max speed": "Max (PPO)",
    "PPO max speed lookahead": "Max lookahead (PPO)",
    "PPO max speed lookahead speed": "Max lookahead speed (PPO)",
    "PPO max speed lookahead road": "Max lookahead road (PPO)",
    "PPO smooth": "Smooth (PPO)",
    "PPO smooth lookahead": "Smooth lookahead (PPO)",
    "DDPG simple": "DDPG",

    # gamma_smooth
    "PPO smooth gamma_smooth=0.1": "Smooth $\gamma_{smooth}=0.1$",
    "PPO smooth gamma_smooth=0.5": "Smooth $\gamma_{smooth}=0.5$",
    "PPO smooth gamma_smooth=1.0": "Smooth $\gamma_{smooth}=1.0$",

    # gamma_acc
    "PPO smooth gamma_acc=0.1": "Smooth $\gamma_{acc}=0.1$",
    "PPO smooth gamma_acc=0.5": "Smooth $\gamma_{acc}=0.5$",
    "PPO smooth gamma_acc=1.0": "Smooth $\gamma_{acc}=1.0$",

    # gamma_smooth
    "PPO max speed gamma_min=0.1": "Max speed $\gamma_{min}=0.1$",
    "PPO max speed gamma_min=1.0": "Max speed $\gamma_{min}=1.0$",
    "PPO max speed gamma_min=10.0": "Max speed $\gamma_{min}=10.0$",

    # MPC
    "Base": "Base",
    "Horizon 5": "Horizon 5",
    "Horizon 10": "Horizon 10",
    "Horizon 20": "Horizon 20",
    "Lower other terms": "Focus on Qv and Qd",
    "Deactivate other terms": "Only Qv and Qd",
    "Deactivate R": "Deactivate R",
    "Increase R": "Increase R",
    "Set linear 1.0": "Linear 1.0",
    "Set linear -1.0": "Linear -1.0",
    "Soft con 0.0": "Soft con. 0.0",
    "Soft con 1.0": "Soft con. 1.0",
    "Soft con 5.0": "Soft con. 5.0",

    # PID
    "(0.06, 0.04, 0.03), (0.05, 0.05, 0.05)": "(0.06, 0.04, 0.03), (0.05, 0.05, 0.05)",
    "(0.1, 0.04, 0.03), (0.1, 0.05, 0.05)": "(0.1, 0.04, 0.03), (0.1, 0.05, 0.05)",
    "(0.06, 0.1, 0.03), (0.05, 0.1, 0.05)": "(0.06, 0.1, 0.03), (0.05, 0.1, 0.05)",
    "(0.06, 0.04, 0.1), (0.05, 0.05, 0.1)": "(0.06, 0.04, 0.1), (0.05, 0.05, 0.1)",
}


LEGEND_MODELS_PATH = {
    # RL
    "PPO simple": "Simple",
    "PPO simple lookahead": "Simple (la)",
    "PPO simple sparse": "Simple (sp)",
    "PPO max speed": "Max",
    "PPO max speed lookahead": "Max (la)",
    "PPO max speed lookahead speed": "Max (la-speed)",
    "PPO max speed lookahead road": "Max (la-road)",
    "PPO smooth": "Smooth",
    "PPO smooth lookahead": "Smooth (la)",
    "DDPG simple": "DDPG",

    # gamma_smooth
    "PPO smooth gamma_smooth=0.1": "Smooth $\gamma_{smooth}=0.1$",
    "PPO smooth gamma_smooth=0.5": "Smooth $\gamma_{smooth}=0.5$",
    "PPO smooth gamma_smooth=1.0": "Smooth $\gamma_{smooth}=1.0$",

    # gamma_acc
    "PPO smooth gamma_acc=0.1": "Smooth $\gamma_{acc}=0.1$",
    "PPO smooth gamma_acc=0.5": "Smooth $\gamma_{acc}=0.5$",
    "PPO smooth gamma_acc=1.0": "Smooth $\gamma_{acc}=1.0$",

    # gamma_smooth
    "PPO max speed gamma_min=0.1": "Max speed $\gamma_{min}=0.1$",
    "PPO max speed gamma_min=1.0": "Max speed $\gamma_{min}=1.0$",
    "PPO max speed gamma_min=10.0": "Max speed $\gamma_{min}=10.0$",

    # MPC
    "Base": "Base",
    "Horizon 5": "Horizon 5",
    "Horizon 10": "Horizon 10",
    "Horizon 20": "Horizon 20",
    "Lower other terms": "Focus Qv, Qd",
    "Deactivate other terms": "Only Qv, Qd",
    "Deactivate R": "No R",
    "Increase R": "Large R",
    "Set linear 1.0": "Linear 1.0",
    "Set linear -1.0": "Linear -1.0",
    "Soft con 0.0": "Soft con. 0.0",
    "Soft con 1.0": "Soft con. 1.0",
    "Soft con 5.0": "Soft con. 5.0",

    # PID
    "(0.06, 0.04, 0.03), (0.05, 0.05, 0.05)": "Set 1",
    "(0.1, 0.04, 0.03), (0.1, 0.05, 0.05)": "Set 2",
    "(0.06, 0.1, 0.03), (0.05, 0.1, 0.05)": "Set 3",
    "(0.06, 0.04, 0.1), (0.05, 0.05, 0.1)": "Set 4",
}



def plot_rewards(output_path=None):
    if output_path is None:
        output_path = PATH

    # Font sizes
    title_fs = 24
    label_fs = 22
    tick_fs = 16
    legend_fs = 18

    # Data ranges
    d = np.linspace(-2, 2, 200)
    theta = np.linspace(-1, 1, 200)
    v_tgt = 5
    v = np.linspace(0, 2*v_tgt, 400)
    v1 = np.linspace(0, 2*v_tgt, 100)
    v2 = np.linspace(0, 2*v_tgt, 100)

    # Create figure with 1 row and 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), subplot_kw={'projection': None})

    # --- First: r_lane_simple (2D heatmap) ---
    D, T = np.meshgrid(d, theta)
    Z1 = r_lane_simple(D, T)
    c1 = axes[0].contourf(D, T, Z1, levels=50)
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("$r_{simple, lane}$(d, θ)", fontsize=title_fs)
    axes[0].set_xlabel("d", fontsize=label_fs)
    axes[0].set_ylabel("θ", fontsize=label_fs)
    axes[0].tick_params(axis='both', labelsize=tick_fs)

    # --- Second: r_speed_simple (2D line) ---
    Z2 = [r_speed_simple(val, v_tgt) for val in v]
    axes[1].plot(v, Z2)
    axes[1].axvline(v_tgt, color="red", linestyle="--")
    axes[1].set_title("$r_{simple, speed}$($v$, $v_{rec}$=" + f"{v_tgt})", fontsize=title_fs)
    axes[1].set_xlabel("v", fontsize=label_fs)
    axes[1].set_ylabel("Reward", fontsize=label_fs)
    axes[1].tick_params(axis='both', labelsize=tick_fs)
    axes[1].set_ylim(0, 1)

    # --- Third: r_speed_max (2D line, multiple gammas) ---
    gammas = [0.1, 1.0, 10.0]
    for gamma in gammas:
        Z3 = [r_speed_max(val, v_tgt, gamma) for val in v]
        axes[2].plot(v, Z3, label=f"γ={gamma}")
    axes[2].axvline(v_tgt, color="red", linestyle="--")
    axes[2].set_title("$r_{max, speed}$($v$, $v_{rec}$=" + f"{v_tgt}, " + r"$\gamma_{min}$)", fontsize=title_fs)
    axes[2].set_xlabel("v", fontsize=label_fs)
    axes[2].set_ylabel("Reward", fontsize=label_fs)
    axes[2].tick_params(axis='both', labelsize=tick_fs)
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=legend_fs)

    # --- Fourth: r_speed_smooth (2D plot, fix v1 and v2) ---
    fixed_pairs = [(v_tgt, v_tgt), (4, v_tgt), (v_tgt, 4)]

    for v1_fixed, v2_fixed in fixed_pairs:
        dv = [r_speed_smooth(val, v1_fixed, v2_fixed) for val in v]
        axes[3].plot(v, dv, label=fr"$x_{{-1}}={v1_fixed}, x_{{-2}}={v2_fixed}$")

    axes[3].set_title(r"$r_{derivative}(x, x_{-1}, x_{-2})$", fontsize=title_fs)
    axes[3].set_xlabel("$x$", fontsize=label_fs)
    axes[3].set_ylabel("Reward", fontsize=label_fs)
    axes[3].tick_params(axis='both', labelsize=tick_fs)
    axes[3].set_ylim(0, 1)
    axes[3].legend(fontsize=legend_fs)

    plt.tight_layout()

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(os.path.join(output_path, "rewards.pdf"), bbox_inches='tight')



def plot_paths(pm_per_sim_lists, models, method_name="", output_path=None, output_name=None):
    if output_path is None:
        output_path = os.path.join(PATH, method_name)
    if output_name is not None:
        output_dir = os.path.join(output_path, output_name)
    else:
        output_dir = output_path

    # Font sizes
    label_fs = 24

    n_roads = len(road_elements_lists)
    n_models = len(models)

    row_labels = [LEGEND_MODELS_PATH[model["name"]] for model in models]
    col_labels = [f"Road: {name}" for name in road_elements_names]

    fig, axs = fig, axs = plt.subplots(n_models, n_roads, sharex=False, sharey=False, figsize=(20, 20))

    if n_roads == 1:
        axs = np.expand_dims(axs, axis=1)

    if n_models == 1:
        axs = np.expand_dims(axs, axis=0)

    # Plot the paths
    for i in range(n_roads):
        for j in range(n_models):
            ax = axs[j, i]

            path = [
                el["paths"] for el in pm_per_sim_lists if 
                    el["model_name"] == models[j]["name"] and 
                    el["road_elements_name"] == road_elements_names[i]
            ][0]

            mean_x = np.array(path["mean_x"])
            mean_y = np.array(path["mean_y"])
            std_x = np.array(path["std_x"])
            std_y = np.array(path["std_y"])

            # Compute radial std (Euclidean)
            radial_std = np.sqrt(std_x**2 + std_y**2)

            # Compute tangent and normal directions
            dx = np.gradient(mean_x)
            dy = np.gradient(mean_y)
            norm = np.sqrt(dx**2 + dy**2)
            norm[norm < 1e-6] = 1e-6   # avoid division by zero
            dx /= norm
            dy /= norm

            # Normal vectors
            nx = -dy
            ny = dx
            
            upper = np.column_stack([mean_x + nx * radial_std, mean_y + ny * radial_std])
            lower = np.column_stack([mean_x - nx * radial_std, mean_y - ny * radial_std])[::-1]

            polygon = np.vstack([upper, lower])  # (2N, 2)

            poly = PolyCollection(
                [polygon],
                facecolor="blue",
                alpha=0.2,
                edgecolor="none",
                joinstyle="round",
                zorder=1
            )
            ax.add_collection(poly)

            # Plot mean trajectory
            ax.plot(mean_x, mean_y, color="blue", linewidth=2, zorder=2)

            # Plot start of road
            ax.scatter(mean_x[0], mean_y[0], color="red", marker="^", s=200, zorder=3, edgecolor='k')

            # Plot the road 
            road = Road(road_elements_list=road_elements_lists[i])
            road.render_road(ax)

            ax.set_xticks([])
            ax.set_yticks([])

            # Set row and column labels
            if i == 0:
                axs[j, i].set_ylabel(row_labels[j], fontsize=label_fs)
            if j == 0:
                axs[j, i].set_title(col_labels[i], fontsize=label_fs)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    if output_name is None:
        name = f"{method_name}_paths_{pd.Timestamp.now().strftime("%m_%d_%H_%M")}.pdf"
    else:
        name = f"{method_name}_paths_{output_name}.pdf"
    fig.savefig(os.path.join(output_dir, name), bbox_inches='tight')



def plot_performance_metrics(pm_per_sim_lists, models, method_name="", output_path=None, output_name=None):
    if output_path is None:
        output_path = os.path.join(PATH, method_name)
    if output_name is not None:
        output_dir = os.path.join(output_path, output_name)
    else:
        output_dir = output_path

    # Font sizes
    title_fs = 24
    label_fs = 24
    tick_fs = 18
    legend_fs = 24

    n_roads = len(road_elements_lists)

    row_labels = ["R1: Speed Limit", "R2: Speed", "R3: Lane Keeping", "R3: Smoothness", "R4: Smoothness"]
    col_labels = [f"Road: {name}" for name in road_elements_names]

    colors = [
        "green",
        "orange",
        "red",
        "blue",
        "purple",
        "black",
        "cyan",
        "magenta",
        "brown"
    ]

    fig, axs = fig, axs = plt.subplots(5, n_roads, sharex=False, sharey=False, figsize=(20, 20))

    if n_roads == 1:
        axs = np.expand_dims(axs, axis=1)

    for i in range(n_roads):
        data = [pm_per_sim_list["performance_metrics"] for pm_per_sim_list in pm_per_sim_lists if pm_per_sim_list["road_elements_name"] == road_elements_names[i]]

        for j, key in enumerate(["r1", "r2", "r3.1", "r3.2", "r4"]):
            pm_values = [[pm_dict[key] for pm_dict in pm_per_sim_list] for pm_per_sim_list in data]

            avgs = [np.mean(pm_value) for pm_value in pm_values]
            stds = [np.std(pm_value) for pm_value in pm_values]

            for k, (avg, std) in enumerate(zip(avgs, stds)):
                # Bar
                axs[j, i].add_patch(plt.Rectangle((k-0.3, 0), 0.6, avg, color=colors[k], alpha=0.7))
                # Whiskers
                axs[j, i].plot([k, k], [avg-std, avg+std], color='black', linewidth=2)
                axs[j, i].plot([k-0.15, k+0.15], [avg-std, avg-std], color='black', linewidth=2)
                axs[j, i].plot([k-0.15, k+0.15], [avg+std, avg+std], color='black', linewidth=2)
                # Average line
                axs[j, i].plot([k-0.3, k+0.3], [avg, avg], color='black', linewidth=2)

                axs[j, i].set_xticks([])

            # Set row and column labels
            if i == 0:
                axs[j, i].set_ylabel(row_labels[j], fontsize=label_fs)
            if j == 0:
                axs[j, i].set_title(col_labels[i], fontsize=label_fs)

            # Ensure y-axis starts at 0
            axs[j, i].set_ylim(bottom=0)

    # Add a legend    
    legend_labels = [LEGEND_MODELS_PM[model["name"]] for model in models]
    legend_handles = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(models))]

    fig.subplots_adjust(right=0.80)
    fig.legend(
        handles=legend_handles,
        title=LEGEND_TITLES_PM[method_name],
        fontsize=legend_fs,
        title_fontsize=title_fs,
        loc='center left',
        bbox_to_anchor=(0.82, 0.5)
    )

    os.makedirs(output_dir, exist_ok=True)
    if output_name is None:
        name = f"{method_name}_metrics_{pd.Timestamp.now().strftime("%m_%d_%H_%M")}.pdf"
    else:
        name = f"{method_name}_metrics_{output_name}.pdf"
    fig.savefig(os.path.join(output_dir, name), bbox_inches='tight')



def plot_performance_metrics_table(pm_per_sim_lists, models, method_name="", output_path=None, output_name=None):
    if output_path is None:
        output_path = os.path.join(PATH, method_name)
    if output_name is not None:
        output_dir = os.path.join(output_path, output_name)
    else:
        output_dir = output_path

    road_names = road_elements_names
    metric_keys = ["r1", "r2", "r3.1", "r3.2", "r4"]
    metric_titles = ["R1: Speed Limit", "R2: Speed", "R3: Lane Keeping", "R3: Smoothness", "R4: Smoothness"]

    table_rows = []
    total_cols = len(road_names) + 1  # roads + mean

    for key, title in zip(metric_keys, metric_titles):
        rows = []
        for model in models:
            model_name = LEGEND_MODELS_PM[model["name"]]
            per_road_means = []
            for road in road_names:
                data = [pm_per_sim_list["performance_metrics"]
                        for pm_per_sim_list in pm_per_sim_lists
                        if pm_per_sim_list["road_elements_name"] == road and
                           pm_per_sim_list["model_name"] == model["name"]]
                if len(data) == 0:
                    per_road_means.append(np.nan)
                else:
                    vals = [pm[key] for run in data for pm in run]
                    per_road_means.append(np.mean(vals))
            mean_val = np.nanmean(per_road_means)
            rows.append([model_name] + per_road_means + [mean_val])

        # Bold minimums per column
        arr = np.array([r[1:] for r in rows], dtype=float)
        formatted_rows = []
        for i, row in enumerate(rows):
            formatted = [row[0]]
            for j, val in enumerate(row[1:]):
                if np.isclose(val, np.nanmin(arr[:, j])):
                    formatted.append(f"\\textbf{{{val:.3f}}}")
                else:
                    formatted.append(f"{val:.3f}")
            formatted_rows.append(formatted)

        table_rows.append((title, formatted_rows))

    # Build LaTeX table inside table environment
    latex_lines = [
        "\\begin{table}[h!]",
        "    \\centering",
        f"    \\begin{{tabular}}{{l{'c' * total_cols}}}",
        "Policy & " + " & ".join(road_names) + " & Mean \\\\",
        "    \\hline"
    ]

    for title, rows in table_rows:
        latex_lines.append(f"    \\multicolumn{{{total_cols + 1}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        for row in rows:
            latex_lines.append("    " + " & ".join(row) + " \\\\")
        latex_lines.append("    \\hline")

    latex_lines.append("    \\end{tabular}")
    latex_lines.append(f"    \\caption{{Summary of the observation parameters for method {method_name}.}}")
    latex_lines.append("    \\label{tab:performance_metrics}")
    latex_lines.append("\\end{table}")

    # Save LaTeX file
    os.makedirs(output_dir, exist_ok=True)
    if output_name is None:
        filename = f"{method_name}_metrics_{pd.Timestamp.now().strftime('%m_%d_%H_%M')}.tex"
    else:
        filename = f"{method_name}_metrics_{output_name}.tex"

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(latex_lines))
