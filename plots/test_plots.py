import os
import json
import random
import matplotlib.pyplot as plt


def read_test_results(tests):
    all_results = []

    for t in tests:
        script_directory = os.path.dirname(os.path.realpath(__file__))
        test_file_name = os.path.join(
            script_directory, '..', 'results', t, 'results', 'test_results.json')
        print(test_file_name)
        if os.path.exists(test_file_name):
            with open(test_file_name, 'r') as file:
                try:
                    data = json.load(file)[0]
                    all_results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {test_file_name}: {e}")
        else:
            print(f"Test results for {t} don't exist")
    return all_results


def create_plots(metrics, data, models_name, configuration, save_plot_prefix="plot"):
    script_directory = os.path.dirname(__file__)
    values = [[d[metric[0]] for d in data] for metric in metrics]
    colors = ["#7b00c2", "#189e00", "#03adfc"]

    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(6, 6))
        # Generate random colors for each bar
        ax.bar(range(len(data)), values[i], color=colors)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([f"{models_name[j]}" for j in range(len(data))])
        ax.set_ylabel(metric[1])
        ax.set_title(f'{metric[1]} Test Results', fontsize=16)

        # Add a description under the title
        ax.text(0.5, -0.1, configuration, ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='black')

        # Save the plot to a file
        save_path = os.path.join(
            script_directory, f"{save_plot_prefix}_{metric[0]}.png")
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")


# ---CONFIGURATIONS---#
test_folders = [
    "2. densenet121_2023-12-19_12-10-36",
    "14. pretrained_2023-12-20_08-28-35",
    "8.standard_2023-12-20_16-28-57"
]

metrics = [tuple(('test_accuracy', 'Accuracy')), tuple(
    ('test_recall', "Recall")), tuple(('test_loss', 'Loss'))]
models_name = ["DenseNet121", "ViT Pretrained", "ViT Standard"]
configuration = "Multiple Loss=True, Segmentation=Dynamic, Keep_Background=True"


assert len(test_folders) == len(
    models_name), "The number of tests and their name must be of equal lenght"

data = read_test_results(test_folders)
create_plots(metrics, data, models_name, configuration)
