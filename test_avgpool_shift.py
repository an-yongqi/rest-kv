import numpy as np

data = np.array([
    [11.58, 20.76, 26.09, 24.06, 23.36, 10.62, 17.49, 17.01, 20.12, 60.5, 78.2, 37.45, 1.83, 25.17, 49.88, 52.64],
    [12.70, 29.01, 38.66, 28.71, 25.10, 14.19, 17.53, 17.70, 20.09, 60.94, 80.75, 35.24, 3.15, 34.02, 47.44, 49.01],
    [13.23, 29.30, 39.32, 29.63, 25.57, 14.55, 18.12, 18.32, 21.01, 61.13, 81.49, 35.78, 3.61, 34.46, 48.39, 50.00],
    [13.36, 29.43, 39.8, 30.24, 26.01, 14.82, 18.3, 18.86, 21.23, 62, 81.51, 36.04, 4.33, 35.08, 49.16, 50.73],
    [13.7, 30.33, 42.08, 30.13, 26.06, 14.37, 18.82, 18.6, 22.38, 69, 81.72, 37.55, 4.33, 35.21, 48.67, 49.48]
])



# Get max and second-max values per column
def get_top_two_values(column):
    sorted_values = sorted(set(column), reverse=True)
    max_value = sorted_values[0]
    second_max_value = sorted_values[1] if len(sorted_values) > 1 else max_value
    return max_value, second_max_value

output_lines = []

# Method names
methods = ['StreamingLLM', 'H2O', 'TOVA', 'SnapKV', '\\method{}']

# Process each row and compute averages
for row, method in zip(data, methods):
    output_row = [method]

    max_value_row, second_max_value_row = get_top_two_values(row)

    for i in range(len(row)):
        max_value, second_max_value = get_top_two_values(data[:, i])
        if row[i] == max_value:
            output_row.append(f"\\textbf{{{row[i]:.2f}}}")
        elif row[i] == second_max_value:
            output_row.append(f"\\underline{{{row[i]:.2f}}}")
        else:
            output_row.append(f"{row[i]:.2f}")
    
    avg_value = np.mean(row)

    # Format the row average
    if avg_value == max_value_row:
        output_row.append(f"\\textbf{{{avg_value:.2f}}}")
    elif avg_value == second_max_value_row:
        output_row.append(f"\\underline{{{avg_value:.2f}}}")
    else:
        output_row.append(f"{avg_value:.2f}")
    
    output_lines.append(" & ".join(output_row))

# Print final results
for line in output_lines:
    print(f"{line} \\\\")
