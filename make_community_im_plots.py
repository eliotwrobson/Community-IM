import json

import matplotlib.pyplot as plt

RESULT_FILE_NAME = "benchmark_result.json"


def get_runtime(
    graph: str,
    weighting_scheme: str,
    use_diffusion_degree: bool,
    algorithm: str,
    budget: int,
    partition_method: str | None,
    resolution_parameter: float | None,
    marginal_gain_error: float | None = None,
) -> tuple[float, float]:
    # Read file and append results
    with open(RESULT_FILE_NAME, "r") as f:
        data = json.load(f)

    for result in data["results"]:
        if (
            result["graph"] == graph
            and result["weighting scheme"] == weighting_scheme
            and result["use diffusion degree"] == use_diffusion_degree
            and result["algorithm"] == algorithm
            and result["budget"] == budget
            and result["partition method"] == partition_method
            and result["resolution parameter"] == resolution_parameter
            and result["marginal gain error"] == marginal_gain_error
        ):
            return result["time taken"], result["influence"]

    raise Exception("Desired result not found")


def main() -> None:
    GRAPH_NAME = "youtube"
    WEIGHTING_SCHEME = "weighted_cascade"

    community_im_influences = []
    community_im_dd_influences = []
    celf_pp_influences = []
    celf_bb_influences = []
    degree_influences = []
    degree_discount_influences = []

    budgets = [5, 10, 20, 50, 100]

    community_im_runtimes = []
    community_im_dd_runtimes = []
    celf_pp_runtimes = []
    celf_bb_runtimes = []
    degree_runtimes = []
    degree_discount_runtimes = []

    for budget in budgets:
        community_im_runtime, community_im_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=False,
            algorithm="community-im",
            budget=budget,
            partition_method="RBConfigurationVertexPartition",
            resolution_parameter=1.0,
            marginal_gain_error=1.0,
        )
        community_im_influences.append(community_im_influence)
        community_im_runtimes.append(community_im_runtime)

        community_im_dd_runtime, community_im_dd_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=True,
            algorithm="community-im",
            budget=budget,
            partition_method="RBConfigurationVertexPartition",
            resolution_parameter=1.0,
            marginal_gain_error=1.0,
        )
        community_im_dd_influences.append(community_im_dd_influence)
        community_im_dd_runtimes.append(community_im_dd_runtime)

        celf_pp_runtime, celf_pp_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=False,
            algorithm="celf-pp",
            budget=budget,
            partition_method=None,
            resolution_parameter=None,
            marginal_gain_error=1.0,
        )
        celf_pp_influences.append(celf_pp_influence)
        celf_pp_runtimes.append(celf_pp_runtime)

        celf_bb_runtime, celf_bb_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=False,
            algorithm="celf-pp",
            budget=budget,
            partition_method=None,
            resolution_parameter=None,
            marginal_gain_error=1.2,
        )
        celf_bb_influences.append(celf_bb_influence)
        celf_bb_runtimes.append(celf_bb_runtime)

        degree_runtime, degree_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=False,
            algorithm="degree",
            budget=budget,
            partition_method=None,
            resolution_parameter=None,
        )
        degree_influences.append(degree_influence)
        degree_runtimes.append(degree_runtime)

        degree_discount_runtime, degree_discount_influence = get_runtime(
            GRAPH_NAME,
            WEIGHTING_SCHEME,
            use_diffusion_degree=False,
            algorithm="degree-discount",
            budget=budget,
            partition_method=None,
            resolution_parameter=None,
        )
        degree_discount_influences.append(degree_discount_influence)
        degree_discount_runtimes.append(degree_discount_runtime)

    # print(
    #     {
    #         "Budgets": budgets,
    #         "Community IM Influences": community_im_influences,
    #         "Community IM DD Influences": community_im_dd_influences,
    #         "CELF++ Influences": celf_pp_influences,
    #         "Degree Influences": degree_influences,
    #         "Degree Discount Influences": degree_discount_influences,
    #     }
    # )

    # print(
    #     {
    #         "Budgets": budgets,
    #         "Community IM Runtimes": community_im_runtimes,
    #         "Community IM DD Runtimes": community_im_dd_runtimes,
    #         "CELF++ Runtimes": celf_pp_runtimes,
    #         "Degree Runtimes": degree_runtimes,
    #         "Degree Discount Runtimes": degree_discount_runtimes,
    #     }
    # )

    (
        community_im_line,
        community_im_dd_line,
        celf_pp_line,
        celf_bb_line,
        degree_line,
        degree_discount_line,
    ) = plt.plot(
        budgets,
        community_im_influences,
        budgets,
        community_im_dd_influences,
        budgets,
        celf_pp_influences,
        budgets,
        celf_bb_influences,
        budgets,
        degree_influences,
        budgets,
        degree_discount_influences,
        markersize=2,
    )

    community_im_line.set_label("Community IM")
    community_im_dd_line.set_label("Community IM DD")
    celf_pp_line.set_label("CELF++")
    celf_bb_line.set_label("CELF||")
    degree_line.set_label("Degree")
    degree_discount_line.set_label("Degree Discount")

    plt.ylabel("Influence")
    plt.xlabel("Budget")
    plt.title(
        f"Influence Maximization on {GRAPH_NAME} Graph with weighting scheme {WEIGHTING_SCHEME}"
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    (
        community_im_runtime_line,
        community_im_dd_runtime_line,
        celf_pp_runtime_line,
        celf_bb_runtime_line,
        degree_runtime_line,
        degree_discount_runtime_line,
    ) = plt.plot(
        budgets,
        community_im_runtimes,
        budgets,
        community_im_dd_runtimes,
        budgets,
        celf_pp_runtimes,
        budgets,
        celf_bb_runtimes,
        budgets,
        degree_runtimes,
        budgets,
        degree_discount_runtimes,
        markersize=2,
    )

    community_im_runtime_line.set_label("Community IM Runtime")
    community_im_dd_runtime_line.set_label("Community IM DD Runtime")
    celf_pp_runtime_line.set_label("CELF++ Runtime")
    celf_bb_runtime_line.set_label("CELF|| Runtime")
    degree_runtime_line.set_label("Degree Runtime")
    degree_discount_runtime_line.set_label("Degree Discount Runtime")

    plt.ylabel("Runtime (s)")
    plt.xlabel("Budget")
    plt.title(
        f"Influence Maximization {GRAPH_NAME} Runtimes with weighting scheme {WEIGHTING_SCHEME}"
    )
    # plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
