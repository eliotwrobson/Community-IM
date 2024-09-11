import csv

import matplotlib.pyplot as plt


def main() -> None:
    graph_name = "Deezer"
    lim_influences = []
    mle_influences = []
    budgets = []

    lim_runtimes = []
    mle_runtimes = []

    with open(f"{graph_name.lower()}_benchmark_results.csv", newline="") as csvfile:
        row_iter = csv.reader(csvfile)
        next(row_iter)
        for row in row_iter:
            lim_influences.append(float(row[1]))
            mle_influences.append(float(row[3]))
            budgets.append(float(row[-1]))

            lim_runtimes.append(float(row[2]))
            mle_runtimes.append(float(row[4]))

    lim_line, mle_line = plt.plot(
        budgets, lim_influences, "o", budgets, mle_influences, "o", markersize=2
    )

    lim_line.set_label("LIM Influence")
    mle_line.set_label("MLE Influence")

    plt.ylabel("Influence")
    plt.xlabel("Budget")
    plt.title(f"LIM vs MLE {graph_name} Influence")
    plt.grid(True)
    plt.legend()
    plt.show()

    lim_line, mle_line = plt.plot(
        budgets, lim_runtimes, "o", budgets, mle_runtimes, "o", markersize=2
    )

    lim_line.set_label("LIM Runtime")
    mle_line.set_label("MLE Runtime")

    plt.ylabel("Runtime (s)")
    plt.xlabel("Budget")
    plt.title(f"LIM vs MLE {graph_name} Runtime")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
