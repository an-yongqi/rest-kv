import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import glob
import os

# Dictionary mapping model_name to pretrained_len
MODEL_PRETRAINED_LENS = {
    "Mistral-7B-Instruct-v0.3": 32768,
    "Llama-3.1-8B-Instruct": 131072,
    "Llama-2-7B-Chat": 4096,
}

def main(args):
    # Retrieve parameters from the parsed arguments
    folder_path = args.folder_path
    model_name = args.model_name
    PRETRAINED_LEN = MODEL_PRETRAINED_LENS[model_name]

    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields

            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(19, 8))  # Adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        linewidths=0.5,  # Adjust the thickness of the grid lines
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--',
        cbar=False
    )

    # More aesthetics
    plt.title(f'{model_name} True Average Score={df["Score"].mean():.2f}', fontsize=24)
    plt.xlabel('Token Limit', fontsize=18)
    plt.ylabel('Depth Percent', fontsize=18)
    xtick_positions = list(range(2000, 32001, 400))
    xtick_labels = [str(pos) if pos % 2000 == 0 else None for pos in xtick_positions]
    plt.xticks(ticks=list(range(76)), labels=xtick_labels, rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    # Extract folder name from folder_path (this assumes the folder name is at the end of the path)
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # Lowercase and strip trailing _False
    folder_name = folder_name.lower().replace('_false', '')

    # Construct the save path based on the folder name
    save_path = f"./results_needle/img/{folder_name}.pdf"
    print(f"saving at {save_path}")
    plt.savefig(save_path, dpi=300)



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate heatmap for model performance based on context length and depth.")

    # Define arguments
    parser.add_argument(
        '--folder_path',
        type=str,
        default='./results_needle/results/Mistral_snapkv_64_reproduce/',
        help='Path to the folder containing JSON results'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Mistral-7B-Instruct-v0.3',
        help='Name of the model'
    )
    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
