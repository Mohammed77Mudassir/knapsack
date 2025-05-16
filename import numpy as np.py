import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ----------- Input Section -------------
n = int(input("Enter number of items: "))
weights = []
values = []

for i in range(n):
    weights.append(int(input(f"Enter weight of item {i+1}: ")))
    values.append(int(input(f"Enter value of item {i+1}: ")))

capacity = int(input("Enter knapsack capacity: "))

# ----------- DP Table Logic (based on your uploaded image) -------------
dp = np.zeros((n + 1, capacity + 1), dtype=int)
states = []

for i in range(n + 1):
    for w in range(capacity + 1):
        if i == 0 or w == 0:
            dp[i][w] = 0
            used = False  # Fix: ensure 'used' is always defined
        elif weights[i - 1] <= w:
            include = values[i - 1] + dp[i - 1][w - weights[i - 1]]
            exclude = dp[i - 1][w]
            dp[i][w] = max(include, exclude)
            used = include > exclude
        else:
            dp[i][w] = dp[i - 1][w]
            used = False
        states.append((i, w, dp.copy(), used))

# ----------- Traceback to find selected items -------------
selected_items = []
w = capacity
for i in range(n, 0, -1):
    if dp[i][w] != dp[i - 1][w]:
        selected_items.append(i - 1)
        w -= weights[i - 1]
selected_items.reverse()

# Add final display state
states.append(('final', None, dp.copy(), False))

# ----------- Animation -------------
fig, ax = plt.subplots(figsize=(14, 7))

def update(frame):
    ax.clear()
    i, w, table, used = states[frame]

    if i != 'final':
        ax.set_title(f"Filling dp[{i}][{w}] — {'Using item' if used else 'Skipping item'} {i}", fontsize=14)
        ax.imshow(table, cmap='Blues', vmin=0, vmax=np.max(table))

        for row in range(n + 1):
            for col in range(capacity + 1):
                ax.text(col, row, str(table[row][col]), ha='center', va='center', fontsize=8)

        if i > 0:
            ax.add_patch(plt.Rectangle((w - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))

        ax.set_xticks(range(capacity + 1))
        ax.set_yticks(range(n + 1))
        ax.set_xlabel("Capacity")
        ax.set_ylabel("Items")
        ax.set_xticklabels(range(capacity + 1))
        ax.set_yticklabels(range(n + 1))

        if i > 0:
            if weights[i - 1] <= w:
                formula = f"dp[{i}][{w}] = max(dp[{i-1}][{w}], dp[{i-1}][{w - weights[i-1]}] + {values[i-1]}) = {table[i][w]}"
            else:
                formula = f"dp[{i}][{w}] = dp[{i-1}][{w}] = {table[i][w]}"
            ax.text(capacity + 1.2, 1, formula, fontsize=10, ha='left', va='top', wrap=True)

    else:
        ax.set_title("Final DP Table - Traceback Complete", fontsize=14)
        ax.imshow(table, cmap='plasma', vmin=0, vmax=np.max(table))

        for row in range(n + 1):
            for col in range(capacity + 1):
                ax.text(col, row, str(table[row][col]), ha='center', va='center', fontsize=8)

        ax.set_xticks(range(capacity + 1))
        ax.set_yticks(range(n + 1))
        ax.set_xlabel("Capacity")
        ax.set_ylabel("Items")
        ax.set_xticklabels(range(capacity + 1))
        ax.set_yticklabels(range(n + 1))

        selected_text = '\n'.join([f"Item {i+1} (W={weights[i]}, V={values[i]})" for i in selected_items])
        total_value = table[n][capacity]
        result_text = f"Selected Items:\n{selected_text}\n\nTotal Max Value: {total_value}"
        ax.text(capacity + 1.2, 1, result_text, fontsize=10, ha='left', va='top',
                bbox=dict(facecolor='lightyellow', edgecolor='black'))

# ----------- Show Animation in Colab -------------
anim = FuncAnimation(fig, update, frames=len(states), interval=400)
HTML(anim.to_jshtml())
