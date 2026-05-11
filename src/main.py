import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Import configuration
from src.config import (
    DATA_DIR,
    EXCESS_PERCENT,
    GA_CROSSOVER_PROB,
    GA_GENERATIONS,
    GA_MUTATION_PROB,
    GA_POPULATION_SIZE,
    LOGS_DIR,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    NUM_PRODUCTS,
    RANDOM_SEED,
    REQUIRED_DATA_FILES,
    RESULTS_DIR,
    SALES_DAYS,
    SHORTAGE_PERCENT,
    VISUALIZATIONS_DIR,
    create_directories,
    get_ga_config,
)

# Import components
from src.data_generator.data_generator_main import generate_all_data
from src.engine.analyzer import InventoryAnalyzer
from src.engine.genetic_algorithm import GeneticAlgorithmOptimizer
from src.engine.results_manager import ResultsManager
from src.engine.rule_based import RuleBasedOptimizer
from src.engine.transportation_simplex import TransportationSimplexOptimizer


def setup_directories():
    """Create necessary directories using config."""
    return create_directories()


def run_data_generation(args):
    """Run data generation."""
    print("\n=== DATA GENERATION ===")
    generate_all_data(
        num_products=args.products,
        days=args.days,
        output_dir=args.data_dir,
        random_seed=args.seed,
        min_days=args.min_days,
        max_days=args.max_days,
        excess_percent=args.excess_percent,
        shortage_percent=args.shortage_percent,
    )


def run_analysis(args):
    """Run inventory analysis."""
    print("\n=== INVENTORY ANALYSIS ===")

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=os.path.join(args.data_dir, "sales_data.csv"),
        inventory_path=os.path.join(args.data_dir, "inventory_data.csv"),
        stores_path=os.path.join(args.data_dir, "stores.csv"),
        products_path=os.path.join(args.data_dir, "products.csv"),
    )

    # Analyze data
    analysis_df = analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances(
        min_days=args.min_days, max_days=args.max_days
    )

    # Save analysis results
    analysis_df.to_csv(
        os.path.join(args.results_dir, "inventory_analysis.csv"), index=False
    )
    excess_df.to_csv(
        os.path.join(args.results_dir, "excess_inventory.csv"), index=False
    )
    needed_df.to_csv(
        os.path.join(args.results_dir, "needed_inventory.csv"), index=False
    )

    # Calculate total excess and needed units
    excess_units = excess_df["excess_units"].sum()
    needed_units = needed_df["needed_units"].sum()

    print(f"\nTotal excess units: {excess_units}")
    print(f"Total needed units: {needed_units}")
    print(f"Excess to needed ratio: {excess_units / needed_units:.2f}")

    return analyzer, analysis_df, excess_df, needed_df


def run_rule_based_optimization(analyzer, excess_df, needed_df, args):
    """Run rule-based optimization."""
    print("\n=== RULE-BASED OPTIMIZATION ===")

    # Create optimizer
    optimizer = RuleBasedOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    execution_time = time.time() - start_time
    print(f"Rule-based optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "rule_based_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(
            os.path.join(args.results_dir, "rule_based_impact.csv")
        )

        return transfer_plan, impact_df

    return transfer_plan, None


def run_ga_optimization(analyzer, excess_df, needed_df, args):
    """Run genetic algorithm optimization."""
    print("\n=== GENETIC ALGORITHM OPTIMIZATION ===")

    # Create optimizer
    optimizer = GeneticAlgorithmOptimizer(random_seed=args.seed)

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(
        excess_df,
        needed_df,
        population_size=args.ga_population,
        num_generations=args.ga_generations,
        crossover_prob=args.ga_crossover,
        mutation_prob=args.ga_mutation,
    )

    execution_time = time.time() - start_time
    print(f"Genetic algorithm optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "ga_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(os.path.join(args.results_dir, "ga_impact.csv"))

        return transfer_plan, impact_df

    return transfer_plan, None


def run_transportation_simplex_optimization(analyzer, excess_df, needed_df, args):
    """Run Transportation Simplex optimization (main academic solver)."""
    print("\n=== TRANSPORTATION SIMPLEX OPTIMIZATION ===")

    optimizer = TransportationSimplexOptimizer()
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    start_time = time.time()
    transfer_plan = optimizer.optimize(excess_df, needed_df)
    execution_time = time.time() - start_time
    print(f"Transportation Simplex completed in {execution_time:.2f} seconds")

    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "transportation_simplex_transfers.csv"),
            index=False,
        )
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)
        pd.DataFrame(impact_df).to_csv(
            os.path.join(args.results_dir, "transportation_simplex_impact.csv")
        )
        return transfer_plan, impact_df, optimizer, execution_time

    return transfer_plan, None, optimizer, execution_time


def create_results(analysis_df, results_dict, analyzer, args):
    """Create simplified results: summary and best transfer plan."""
    print("\n=== GENERATING RESULTS ===")

    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))

    results_manager = ResultsManager(args.results_dir)
    # results_dict values may be 2-tuples (plan, impact) or 3/4-tuples;
    # ResultsManager expects {name: (plan, impact)}
    slim_dict = {
        k: (v[0], v[1]) for k, v in results_dict.items()
    }
    results_manager.create_final_results(slim_dict, stores_df, products_df)


def _coverage_rate(transfer_plan: pd.DataFrame, needed_df: pd.DataFrame):
    """
    Compute coverage KPIs correctly:
      covered_units = sum over each (to_store_id, product_id) of
                      min(received_units, needed_units)
      coverage_rate = covered_units / total_needed_units * 100
    """
    total_needed = int(needed_df["needed_units"].sum())
    if total_needed == 0:
        return 0, 100.0, 0

    if transfer_plan is None or transfer_plan.empty:
        return 0, 0.0, total_needed

    received = (
        transfer_plan.groupby(["to_store_id", "product_id"])["units"]
        .sum()
        .reset_index()
        .rename(columns={"units": "received_units"})
    )
    merged = needed_df[["store_id", "product_id", "needed_units"]].merge(
        received,
        left_on=["store_id", "product_id"],
        right_on=["to_store_id", "product_id"],
        how="left",
    )
    merged["received_units"] = merged["received_units"].fillna(0)
    merged["covered"] = merged[["received_units", "needed_units"]].min(axis=1)
    covered = int(merged["covered"].sum())
    unmet = total_needed - covered
    rate = covered / total_needed * 100
    return covered, round(rate, 2), unmet


def create_algorithm_comparison(
    results_dict: dict,
    needed_df: pd.DataFrame,
    results_dir: str,
):
    """
    Write results/algorithm_comparison.csv comparing all algorithms.

    results_dict: {algorithm_name: (transfer_plan, impact_df, optimizer, exec_time)}
    The 3rd and 4th elements are optional (may be None).
    """
    rows = []
    total_needed = int(needed_df["needed_units"].sum())

    for algo, vals in results_dict.items():
        plan = vals[0]
        exec_time = vals[3] if len(vals) > 3 else None
        optimizer = vals[2] if len(vals) > 2 else None

        covered, rate, unmet = _coverage_rate(plan, needed_df)

        if plan is not None and not plan.empty:
            total_cost = plan["transport_cost"].sum()
            n_transfers = len(plan)
            total_units = int(plan["units"].sum())
            avg_cost = total_cost / total_units if total_units > 0 else 0
        else:
            total_cost = n_transfers = total_units = avg_cost = 0

        # Iteration / status from TS solver stats
        solver_status = "N/A"
        iterations = "N/A"
        if optimizer is not None and hasattr(optimizer, "solver_stats"):
            stats = optimizer.solver_stats.get("per_product", [])
            if stats:
                statuses = [s["status"] for s in stats]
                solver_status = "optimal" if all(s == "optimal" for s in statuses) else "mixed"
                iterations = sum(s["iterations"] for s in stats)

        rows.append({
            "algorithm": algo,
            "total_transport_cost": round(total_cost, 2),
            "number_of_transfers": n_transfers,
            "total_units_transferred": total_units,
            "covered_units": covered,
            "total_needed_units": total_needed,
            "coverage_rate": rate,
            "unmet_units": unmet,
            "avg_cost_per_unit": round(avg_cost, 2),
            "execution_time_seconds": round(exec_time, 3) if exec_time else None,
            "solver_status": solver_status,
            "iterations": iterations,
        })

    cmp_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "algorithm_comparison.csv")
    cmp_df.to_csv(out_path, index=False)
    print(f"\nAlgorithm comparison saved to {out_path}")
    print(cmp_df.to_string(index=False))
    return cmp_df


def main():
    parser = argparse.ArgumentParser(
        description="Inventory Transfer Optimization System"
    )

    # General options
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR, help="Results directory"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=VISUALIZATIONS_DIR,
        help="Visualizations directory",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    # Data generation options
    parser.add_argument("--generate-data", action="store_true", help="Generate data")
    parser.add_argument(
        "--products", type=int, default=NUM_PRODUCTS, help="Number of products"
    )
    parser.add_argument(
        "--days", type=int, default=SALES_DAYS, help="Number of days of sales data"
    )
    parser.add_argument(
        "--excess-percent",
        type=int,
        default=EXCESS_PERCENT,
        help="Percentage of items with excess inventory",
    )
    parser.add_argument(
        "--shortage-percent",
        type=int,
        default=SHORTAGE_PERCENT,
        help="Percentage of items with shortage",
    )

    # Analysis options
    parser.add_argument(
        "--min-days",
        type=int,
        default=MIN_INVENTORY_DAYS,
        help="Minimum days of inventory",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=MAX_INVENTORY_DAYS,
        help="Maximum days of inventory",
    )

    # Optimization options
    parser.add_argument(
        "--rule-based", action="store_true", help="Run rule-based optimization"
    )
    parser.add_argument(
        "--ga", action="store_true", help="Run genetic algorithm optimization"
    )
    parser.add_argument(
        "--transportation-simplex", "--ts",
        dest="transportation_simplex",
        action="store_true",
        help="Run Transportation Simplex (main academic solver)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run Rule-Based, Transportation Simplex, and GA (GA is slow ~7 min)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After TS, validate objective with scipy linprog",
    )

    # Genetic algorithm options
    parser.add_argument(
        "--ga-population",
        type=int,
        default=GA_POPULATION_SIZE,
        help="GA population size",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=GA_GENERATIONS,
        help="GA number of generations",
    )
    parser.add_argument(
        "--ga-crossover",
        type=float,
        default=GA_CROSSOVER_PROB,
        help="GA crossover probability",
    )
    parser.add_argument(
        "--ga-mutation",
        type=float,
        default=GA_MUTATION_PROB,
        help="GA mutation probability",
    )

    # Display options
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only summary results"
    )

    args = parser.parse_args()

    # Create directories
    directories = setup_directories()
    args.data_dir = str(directories["data"])
    args.vis_dir = str(directories["visualizations"])
    args.results_dir = str(directories["results"])

    # Generate data if needed
    if args.generate_data:
        run_data_generation(args)

    # Check if data exists
    for file in REQUIRED_DATA_FILES:
        file_path = Path(args.data_dir) / file
        if not file_path.exists():
            print(
                f"Required file {file} not found. Please run with --generate-data first."
            )
            return

    # Run analysis
    analyzer, analysis_df, excess_df, needed_df = run_analysis(args)

    # Run optimizations -- results_dict: {name: (plan, impact, optimizer_or_None, exec_time)}
    results_dict = {}

    if args.rule_based or args.all:
        t0 = time.time()
        transfer_plan, impact_df = run_rule_based_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Rule-Based"] = (transfer_plan, impact_df, None, time.time() - t0)

    if args.transportation_simplex or args.all:
        plan, impact, optimizer, exec_time = run_transportation_simplex_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Transportation-Simplex"] = (plan, impact, optimizer, exec_time)
        if getattr(args, "validate", False) and optimizer is not None:
            optimizer._validate_with_linprog(excess_df, needed_df)

    if args.ga or args.all:
        print("\n[NOTE] GA is slow (~7-8 min). Use --rule-based --ts to skip GA.")
        t0 = time.time()
        transfer_plan, impact_df = run_ga_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Genetic-Algorithm"] = (transfer_plan, impact_df, None, time.time() - t0)

    # Create comprehensive results and reports
    if results_dict:
        create_results(analysis_df, results_dict, analyzer, args)
        create_algorithm_comparison(results_dict, needed_df, args.results_dir)

    print("\n=== INVENTORY TRANSFER OPTIMIZATION COMPLETE ===")
    print(f"Results saved to {args.results_dir} directory:")
    print(f"  * result_summary.txt          - Algorithm comparison")
    print(f"  * best_transfer_plan.csv       - Best algorithm plan")
    print(f"  * algorithm_comparison.csv     - Full KPI comparison table")
    if getattr(args, "transportation_simplex", False) or getattr(args, "all", False):
        print(f"  * transportation_simplex_transfers.csv")
        print(f"  * transportation_simplex_impact.csv")


if __name__ == "__main__":
    main()