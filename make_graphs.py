import csv
import os

import matplotlib.pyplot as plt


def main() -> None:
    result_folder = "benchmark_results"
    graph_name = "wikipedia"
    lim_influences = []
    mle_influences = []
    budgets = []

    # First, get LIM and MLE runtimes
    lim_runtimes = []
    mle_runtimes = []

    with open(
        result_folder + os.sep + f"{graph_name.lower()}_benchmark_results.csv",
        newline="",
    ) as csvfile:
        row_iter = csv.reader(csvfile)
        next(row_iter)
        for row in row_iter:
            lim_influences.append(float(row[1]))
            mle_influences.append(float(row[3]))
            budgets.append(float(row[-1]))

            lim_runtimes.append(float(row[2]))
            mle_runtimes.append(float(row[4]))

    # Next, get the cd runtimes
    cd_influences = []
    cd_runtimes = []
    cd_budgets = []

    with open(
        result_folder + os.sep + f"{graph_name.lower()}_cd_results.csv"
    ) as csvfile:
        row_iter = csv.reader(csvfile)
        next(row_iter)
        for row in row_iter:
            cd_budgets.append(float(row[1]))
            cd_influences.append(float(row[2]))
            cd_runtimes.append(float(row[3]))

    lim_line, mle_line, cd_line = plt.plot(
        budgets,
        lim_influences,
        "o",
        budgets,
        mle_influences,
        "o",
        cd_budgets,
        cd_influences,
        "o",
        markersize=2,
    )

    lim_line.set_label("LIM Influence")
    mle_line.set_label("MLE Influence")
    cd_line.set_label("CD Influence")

    plt.ylabel("Influence")
    plt.xlabel("Budget")
    plt.title(f"LIM vs MLE {graph_name} Influence")
    plt.grid(True)
    plt.legend()
    plt.show()

    lim_line, mle_line, cd_line = plt.plot(
        budgets,
        lim_runtimes,
        "o",
        budgets,
        mle_runtimes,
        "o",
        cd_budgets,
        cd_runtimes,
        "o",
        markersize=2,
    )

    lim_line.set_label("LIM Runtime")
    mle_line.set_label("MLE Runtime")
    cd_line.set_label("CD Runtime")

    plt.ylabel("Runtime (s)")
    plt.xlabel("Budget")
    plt.title(f"LIM vs MLE {graph_name} Runtime")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
