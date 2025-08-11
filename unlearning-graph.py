import re
import matplotlib.pyplot as plt

def parse_log_file(filename):
    rounds = []
    losses = []
    accuracies = []

    pattern = re.compile(r"Round (\d+) - Loss: ([0-9.]+), Accuracy: ([0-9.]+)")

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                round_num = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))

                rounds.append(round_num)
                losses.append(loss)
                accuracies.append(acc)

    return rounds, losses, accuracies

def plot_metrics(rounds, losses, accuracies, unlearn_round=None):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(rounds, losses, 'o-', color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # second y-axis for accuracy
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(rounds, accuracies, 's-', color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    if unlearn_round is not None:
        plt.axvline(x=unlearn_round, color='gray', linestyle='--', label='Unlearning Start')

    plt.title('Loss and Accuracy over Rounds')
    fig.tight_layout()
    plt.legend(loc='upper left')
    plt.show()

# Exemple d’utilisation :
filename = "logs\logunlearn.txt"  # fichier contenant tes logs
rounds, losses, accuracies = parse_log_file(filename)
plot_metrics(rounds, losses, accuracies, unlearn_round=5)
