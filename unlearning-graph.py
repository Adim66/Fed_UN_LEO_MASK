import re
import matplotlib.pyplot as plt

def parse_log_file(filename, start_round=5, end_round=15):
    rounds, losses, accuracies = [], [], []
    print(f"fist line of the file: {filename}est ")
    pattern = re.compile(r"Round (\d+) - Loss: ([0-9.]+), Accuracy: ([0-9.]+)")
    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                round_num = int(match.group(1))
                if start_round <= round_num <= end_round:
                    rounds.append(round_num)
                    losses.append(float(match.group(2)))
                    accuracies.append(float(match.group(3)))
    return rounds, losses, accuracies

def plot_two_curves(rounds, losses, accuracies, unlearn_round=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

    # --- Courbe Loss ---
    ax1.plot(rounds, losses, 'o-', color='tab:red', markersize=8, linewidth=2, label='Loss')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    if unlearn_round is not None:
        ax1.axvline(x=unlearn_round, color='gray', linestyle='--', linewidth=2)
        ax1.text(unlearn_round + 0.1, max(losses)*0.9, 'Unlearning Start', rotation=90, color='gray')
    ax1.legend(fontsize=10)
    ax1.set_title('Loss over Rounds', fontsize=14, fontweight='bold')

    # --- Courbe Accuracy ---
    ax2.plot(rounds, accuracies, 's-', color='tab:blue', markersize=8, linewidth=2, label='Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    if unlearn_round is not None:
        ax2.axvline(x=unlearn_round, color='gray', linestyle='--', linewidth=2)
    ax2.legend(fontsize=10)
    ax2.set_title('Accuracy over Rounds', fontsize=14, fontweight='bold')

    plt.xticks(rounds)
    fig.tight_layout()
    plt.show()

# --- Utilisation ---
filename = "logs\logunlearn.txt"
rounds, losses, accuracies = parse_log_file(filename)
plot_two_curves(rounds, losses, accuracies, unlearn_round=5)
