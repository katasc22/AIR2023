import matplotlib.pyplot as plt

# Sample data for three models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def createmultiPlot(model_names, heading, bar_values, bar_labels, is_horizontal):
    # Generating index for the models
    ind = np.arange(len(model_names))
    print(ind)

    if is_horizontal:
        height = 0.2
        plt.figure(figsize=(6, 6))

        plt.barh(ind - height, bar_values[model_names[0]], height, label=bar_labels[0], color='#f9d448')

        plt.barh(ind, bar_values[model_names[1]], height, label=bar_labels[1], color='#7ca655')

        plt.barh(ind + height, bar_values[model_names[2]], height, label=bar_labels[2], color='#4495a2')

        plt.ylabel('Approaches')
        plt.xlabel('Scores')
        plt.title(heading)
        plt.yticks(ind, model_names)
        plt.xlim(0, 1)
        plt.legend()
        plt.show()
    else:
        width = 0.2
        plt.figure(figsize=(6, 6))

        # Plotting accuracy scores
        plt.bar(ind - width, bar_values[model_names[0]], width, label=bar_labels[0], color='#f9d448')

        plt.bar(ind, bar_values[model_names[1]], width, label=bar_labels[1], color='#7ca655')

        plt.bar(ind + width, bar_values[model_names[2]], width, label=bar_labels[2], color='#4495a2')

        plt.xlabel('Approaches')
        plt.ylabel('Scores')
        plt.title(heading)
        plt.xticks(ind, model_names)
        plt.ylim(0, 1)  # Set the y-axis limit for better visualization of scores (0 to 1)
        plt.legend()
        plt.show()


def createsinglePlot(model_names, heading, bar_values, bar_label, is_horizontal):
    # Generating index for the models
    ind = np.arange(len(model_names))
    print(ind)

    if is_horizontal:
        height = 0.2
        plt.figure(figsize=(6, 6))

        plt.barh(ind, [15.274346590042114, 136.41511583328247, 64.3079993724823], height, color='#f9d448')

        plt.ylabel('Approaches')
        plt.xlabel('Scores')
        plt.title(heading)
        plt.yticks(ind, model_names)
        plt.xlim(0, 200)
        plt.show()


def creatememPlot(model_names, heading, bar_values, bar_labels, is_horizontal, range):
    # Generating index for the models
    ind = np.arange(len(model_names))
    print(ind)

    if is_horizontal:
        height = 0.2
        plt.figure(figsize=(6, 6))

        plt.barh(ind - height, bar_values.iloc[:, 0], height, label=bar_labels[0], color='#f9d448')

        plt.barh(ind, bar_values.iloc[:, 1], height, label=bar_labels[1], color='#7ca655')

        # plt.barh(ind + height, bar_values[model_names[2]], height, label=bar_labels[2], color='#4495a2')

        plt.ylabel('Approaches')
        plt.xlabel('Scores')
        plt.title(heading)
        plt.yticks(ind, model_names)
        plt.xlim(0, range)
        plt.legend()
        plt.show()
    else:
        width = 0.2
        plt.figure(figsize=(6, 6))

        # Plotting accuracy scores
        plt.bar(ind - width, bar_values.iloc[:, 0], width, label=bar_labels[0], color='#f9d448')

        plt.bar(ind, bar_values.iloc[:, 1], width, label=bar_labels[1], color='#7ca655')

        plt.xlabel('Approaches')
        plt.ylabel('Scores')
        plt.title(heading)
        plt.xticks(ind, model_names)
        plt.ylim(0, range)  # Set the y-axis limit for better visualization of scores (0 to 1)
        plt.legend()
        plt.show()


model_names = ['distiluse', 'Mbert', 'minilm']
frame = pd.DataFrame()
frame["distiluse"] = [0.24883313420947822, 0.2860667349377026, 0.0]
frame["Mbert"] = [0.14588240314046766, 0.19983785628946912, 0.0]
frame["minilm"] = [0.5995499123584122, 0.6102597448776406, 0.3076471346168853]
print(frame.head())
frame = frame.T
frame.columns = ['distiluse', 'Mbert', 'minilm']
createmultiPlot(model_names, "MAP-Values", frame, ["multilingual", "monolingual", "not_pretranslated"], False)

runframe = pd.DataFrame()
runframe["multilingual"] = [15.274346590042114, 136.41511583328247, 64.3079993724823]
runframe["monolingual"] = [15, 132.62396754, 17.509822130203247]
creatememPlot(model_names, "Runtime in s", runframe, ["multilingual", "monolingual"], True, 200)

mem_frame = pd.DataFrame()
mem_frame["CPU"] = [2885, 4084, 3089]
mem_frame["CUDA"] = [828, 4394, 1356]
creatememPlot(model_names, "Memory in MB", mem_frame, ["CPU", "CUDA"], False, 4500)
