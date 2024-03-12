import numpy as np
from ergonomics import RULA
#from src.ergonomics_torch import Ergonomics_Torch
import os
import matplotlib.pyplot as plt


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_scores_comparison(scores_initial, scores_optim, fps=None, file_name='comparison_plot'):
    output_directory = 'data/output_graphs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.figure(figsize=(16, 6), dpi=120)

    # Make sure the timestamps array is correctly calculated for the longer of the two score arrays
    max_length = max(len(scores_initial), len(scores_optim))

    if fps is not None:
        # Calculate timestamps to match the length of the longer scores array
        timestamps = np.linspace(0, max_length / fps, max_length)
        if len(scores_initial) == max_length:
            plt.plot(timestamps[:len(scores_initial)], scores_initial, label='Initial', color='b')
            plt.plot(timestamps[:len(scores_optim)], scores_optim, label='Optimized', color='orange')
        else:
            # Adjust the plotting to ensure arrays align
            plt.plot(timestamps[:len(scores_initial)], scores_initial, label='Initial', color='b')
            plt.plot(timestamps[:len(scores_optim)], scores_optim, label='Optimized', color='orange')
        plt.xlabel('Video duration in seconds', fontsize=25, fontweight='medium')
    else:
        plt.plot(range(len(scores_initial)), scores_initial, label='Initial', color='b')
        plt.plot(range(len(scores_optim)), scores_optim, label='Optimized', color='orange')
        plt.xlabel('Video frames', fontsize=25, fontweight='medium')

    # Add risk level guide lines
    plt.plot(np.array([2] * max_length), linestyle='--', color='g')
    plt.plot(np.array([4] * max_length), linestyle='--', color='y')
    plt.plot(np.array([6] * max_length), linestyle='--', color='r')

    plt.ylim([1, 7])
    plt.yticks([1, 2, 3, 4, 5, 6, 7], fontsize=20, fontweight='medium')
    plt.xlim([0, max_length / fps if fps is not None else max_length])
    plt.xticks(fontsize=20, fontweight='medium')
    plt.ylabel('Ergonomic Risk', fontsize=25, fontweight='medium')
    plt.legend()

    RULA_txt = '1-2 = acceptable posture\n3-4 = further investigation, change may be needed\n' \
               '5-6 = further investigation, change soon\n7 = investigate and implement changes'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.5, 0.2, RULA_txt, fontsize=14, horizontalalignment='center', verticalalignment='top', bbox=props)

    plt.subplots_adjust(bottom=0.35)

    save_path = os.path.join(output_directory, f'{file_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

npz_directory = 'data/npz'
optimized_npz_directory = 'data/optimised_npz'

npz_files = os.listdir(npz_directory)

#optim_module = Ergonomics_Torch(device)
#optim_module.to(device)

for file_name in npz_files:

    npz_file_path = os.path.join(npz_directory, file_name)

    # Construct the potential full path to the matching file in the optimised_npz directory
    optimized_file_path = os.path.join(optimized_npz_directory, file_name)

    # Check if the matching file exists in the optimised_npz directory
    if os.path.exists(optimized_file_path):
        print(f"--------------------------------------------------: {file_name}")

        # Load the npz file from the npz directory
        npz_data = np.load(npz_file_path, allow_pickle=True)
        # Generate content summary for the npz file
        npz_content_summary = {key: npz_data[key].shape for key in npz_data}

        rula_eval_initial = RULA(npz_data['kps'])
        scores_initial = rula_eval_initial.compute_scores()
        #rula_eval_initial.plot_scores(fps=30)
        print('1 Initial-Scores: ', np.mean(scores_initial))

        # torch evealuation
        #kps_tensor = torch.from_numpy(npz_data['kps']).to(device)
        #scores_initial_torch = optim_module.compute_scores(kps_tensor)
        #print('2 Initial-torch-Scores: ', scores_initial_torch.detach().cpu().mean().item())

        # Load the npz file from the optimised_npz directory
        optimized_data = np.load(optimized_file_path, allow_pickle=True)

        rula_eval_optim = RULA(optimized_data['kps'])
        scores_optim = rula_eval_optim.compute_scores()
        #scores_optim.plot_scores(fps=30)
        print('3 Optim-Scores: ', np.mean(scores_optim))
        #torch optim evaluation
        #kps_tensor = torch.from_numpy(optimized_data['kps']).to(device)

        #scores_optim_torch = optim_module.compute_scores(kps_tensor)
        plot_scores_comparison(scores_initial, scores_optim, fps=30, file_name=file_name)

    else:
        print(f"No match found for file: {file_name}")
