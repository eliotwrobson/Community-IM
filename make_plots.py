import csv
import os

import matplotlib.pyplot as plt


def main() -> None:
    result_folder = "benchmark_results"
    graph_name = "facebook"
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
            budgets.append(float(row[5]))

            lim_runtimes.append(float(row[2]))
            # mle_runtimes.append(get_mle_runtime(graph_name, budget))
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

    # Next, get the hd runtimes
    hd_influences = []
    hd_runtimes = []
    hd_budgets = []

    with open(
        result_folder + os.sep + f"{graph_name.lower()}_hd_results.csv"
    ) as csvfile:
        row_iter = csv.reader(csvfile)
        next(row_iter)
        for row in row_iter:
            hd_budgets.append(float(row[1]))
            hd_influences.append(float(row[2]))
            hd_runtimes.append(float(row[3]))

    lim_line, mle_line, cd_line, hd_line = plt.plot(
        budgets,
        lim_influences,
        "o",
        budgets,
        mle_influences,
        "o",
        cd_budgets,
        cd_influences,
        "o",
        hd_budgets,
        hd_influences,
        "o",
        markersize=2,
    )

    lim_line.set_label("LIM Influence")
    mle_line.set_label("MLE Influence")
    cd_line.set_label("CD Influence")
    hd_line.set_label("UD Influence")

    plt.ylabel("Influence")
    plt.xlabel("Budget")
    plt.title(f"Influence Maximization on {graph_name} Graph")
    plt.grid(True)
    plt.legend()
    plt.show()

    lim_line, mle_line, cd_line, hd_line = plt.plot(
        budgets,
        lim_runtimes,
        "o",
        budgets,
        mle_runtimes,
        "o",
        cd_budgets,
        cd_runtimes,
        "o",
        hd_budgets,
        hd_runtimes,
        "o",
        markersize=2,
    )

    lim_line.set_label("LIM Runtime")
    mle_line.set_label("MLE Runtime")
    cd_line.set_label("CD Runtime")
    hd_line.set_label("UD Runtime")

    plt.ylabel("Runtime (s)")
    plt.xlabel("Budget")
    plt.title(f"Influence Maximization {graph_name} Runtimes")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
