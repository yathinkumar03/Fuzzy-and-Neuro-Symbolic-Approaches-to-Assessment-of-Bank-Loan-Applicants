
import matplotlib.pyplot as plt

# ==========================================
# MODELS
# ==========================================

models = [
    'FES',
    'NBES'
]

# ==========================================
# ACCURACY VALUES
# ==========================================

accuracy = [
    89.86,   # FES Accuracy
    98.42    # NBES Accuracy
]

# ==========================================
# CREATE BAR GRAPH
# ==========================================

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    models,
    accuracy
)

# ==========================================
# LABELS
# ==========================================

plt.title(
    'Accuracy Comparison between FES and NBES'
)

plt.xlabel(
    'Models'
)

plt.ylabel(
    'Accuracy (%)'
)

plt.ylim(0, 100)

# ==========================================
# DISPLAY VALUES ON BARS
# ==========================================

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()
        + bar.get_width()/2,

        height + 1,

        f'{height}%',

        ha='center'
    )

# ==========================================
# GRID
# ==========================================

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

# ==========================================
# SAVE FIGURE
# ==========================================

plt.savefig(
    "../results/fes_vs_nbes_accuracy.png"
)

# ==========================================
# SHOW GRAPH
# ==========================================

plt.show()

print(
    "\nAccuracy comparison graph saved successfully!"
)

