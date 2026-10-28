import os
import re
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import MultipleLocator, FuncFormatter
from ratingcurve.ratings import PowerLawRating

import rating_curve_DISCHARGE_STAGE_PROBABILITY

def get_stage_flows(
    rating_curve_df,
    max_stage,
    increment=1
):

    # --------------------------------------------------------
    # CLEAN AND SORT RATING CURVE
    # --------------------------------------------------------

    rating_curve = (
        rating_curve_df[["stage_ft", "median_cfs"]]
        .copy()
        .dropna()
        .sort_values("stage_ft")
        .drop_duplicates("stage_ft")
    )

    if len(rating_curve) < 2:
        raise ValueError("At least two rating points are required.")

    if increment <= 0:
        raise ValueError("Increment must be greater than zero.")

    # --------------------------------------------------------
    # AUTOMATIC MINIMUM STAGE
    # --------------------------------------------------------

    min_stage = rating_curve["stage_ft"].min()

    # Round up to nearest increment
    start_stage = np.ceil(min_stage / increment) * increment

    if max_stage > rating_curve["stage_ft"].max():
        raise ValueError(
            "Maximum requested stage exceeds rating curve range."
        )

    if max_stage < start_stage:
        raise ValueError(
            "Maximum stage is below the starting stage."
        )

    # --------------------------------------------------------
    # CREATE STAGE INCREMENTS
    # --------------------------------------------------------

    stages = np.arange(
        start_stage,
        max_stage + increment / 2,
        increment
    )

    # --------------------------------------------------------
    # INTERPOLATE DISCHARGE
    # --------------------------------------------------------

    flows_cfs = np.interp(
        stages,
        rating_curve["stage_ft"],
        rating_curve["median_cfs"]
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DATAFRAME
    # --------------------------------------------------------

    stage_flows = pd.DataFrame({
        "stage_ft": stages,
        "flow_cfs": flows_cfs,
        "flow_cms": flows_cfs * 0.028316846592
    })

    return stage_flows




def get_gssha_rating_curve(
    model_results_dir,
    tail_records=120,
    offset_m=0.0,
    increment=1
):

    model_results_dir = Path(model_results_dir)

    ows_files = {}
    steady_state_WSE = {}
    results = []

    if tail_records <= 0:
        raise ValueError("tail_records must be positive.")

    if increment <= 0:
        raise ValueError("increment must be positive.")

    # ========================================================
    # READ GSSHA FILES
    # ========================================================

    for filepath in sorted(model_results_dir.glob("*.ows")):

        match = re.search(r"m-(\d+)-", filepath.name)

        if match is None:
            continue

        flow_cfs = int(match.group(1))

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=["time", "WSE_m"]
        )

        df["WSE_m"] = pd.to_numeric(
            df["WSE_m"],
            errors="coerce"
        )

        df = df.dropna(subset=["WSE_m"])

        if df.empty:
            continue

        ows_files[filepath.name] = df

        # Mean of final records
        height_m = df["WSE_m"].tail(tail_records).mean()

        steady_state_WSE[filepath.name] = height_m

        # Apply offset
        adjusted_height_m = height_m + offset_m
        stage_ft = adjusted_height_m * 3.280839895

        results.append({
            "flow_cfs": flow_cfs,
            "flow_cms": flow_cfs * 0.028316846592,
            "GSSHA_height_m": height_m,
            "offset_m": offset_m,
            "adjusted_height_m": adjusted_height_m,
            "stage_ft": stage_ft
        })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not results:
        raise ValueError("No valid GSSHA results found.")

    gssha_rating_curve = (
        pd.DataFrame(results)
        .sort_values("flow_cfs")
        .reset_index(drop=True)
    )

    # ========================================================
    # ADD INCREMENTAL STAGE COLUMN
    # NO INTERPOLATION
    # ========================================================

    start_stage = (
        np.ceil(
            gssha_rating_curve["stage_ft"].min() / increment
        ) * increment
    )

    incremental_stages = (
        start_stage
        + np.arange(len(gssha_rating_curve)) * increment
    )

    if float(increment).is_integer():
        incremental_stages = incremental_stages.astype(int)

    gssha_rating_curve["incremental_stage_ft"] = (
        incremental_stages
    )

    return gssha_rating_curve, steady_state_WSE, ows_files

def predict_gssha_probability(
    gssha_rating_curve,
    watershed,
    script_dir=None
):

    # --------------------------------------------------------
    # DIRECTORY
    # --------------------------------------------------------

    if script_dir is None:
        script_dir = Path.cwd()

    script_dir = Path(script_dir)

    probability_dir = (
        script_dir
        / "SAVE_DATA"
        / "MERGED-rating-probability"
    )

    # --------------------------------------------------------
    # FIND WATERSHED CSV
    # --------------------------------------------------------

    matches = [
        p for p in probability_dir.glob("*.csv")
        if p.stem.lower() == (
            f"{watershed}_stage_discharge_probability".lower()
        )
    ]

    if not matches:
        raise FileNotFoundError(
            f"No probability CSV found for {watershed} "
            f"in {probability_dir}"
        )

    probability_file = matches[0]

    print(f"Loading probability: {probability_file.name}")

    # --------------------------------------------------------
    # LOAD PROBABILITY CURVE
    # --------------------------------------------------------

    probability_df = pd.read_csv(probability_file)

    required_columns = [
        "median_cfs",
        "EXC_Prob"
    ]

    for col in required_columns:
        if col not in probability_df.columns:
            raise ValueError(
                f"Missing column in CSV: {col}"
            )

    probability = (
        probability_df[required_columns]
        .copy()
    )

    probability["median_cfs"] = pd.to_numeric(
        probability["median_cfs"],
        errors="coerce"
    )

    probability["EXC_Prob"] = pd.to_numeric(
        probability["EXC_Prob"],
        errors="coerce"
    )

    probability = (
        probability
        .dropna()
        .loc[
            lambda df:
            (df["median_cfs"] >= 0)
            & (df["EXC_Prob"] > 0)
            & (df["EXC_Prob"] <= 1)
        ]
        .sort_values("median_cfs")
        .drop_duplicates("median_cfs")
        .reset_index(drop=True)
    )

    if len(probability) < 2:
        raise ValueError(
            "At least two valid probability points required."
        )

    # --------------------------------------------------------
    # EXTRACT REFERENCE DISCHARGE AND PROBABILITY
    # --------------------------------------------------------

    reference_flow = probability[
        "median_cfs"
    ].to_numpy(dtype=float)

    reference_aep = probability[
        "EXC_Prob"
    ].to_numpy(dtype=float)

    # Check that AEP decreases or remains constant
    # as discharge increases.
    if np.any(np.diff(reference_aep) > 1e-10):
        raise ValueError(
            "Reference probability increases with discharge. "
            "Check the saved CSV before interpolation."
        )

    # --------------------------------------------------------
    # CREATE PCHIP INTERPOLATOR
    # DISCHARGE -> LOG(AEP)
    # --------------------------------------------------------

    probability_interpolator = PchipInterpolator(
        reference_flow,
        np.log(reference_aep),
        extrapolate=False
    )

    # --------------------------------------------------------
    # COPY ORIGINAL GSSHA RESULTS
    # --------------------------------------------------------

    result = gssha_rating_curve.copy()

    if "flow_cfs" not in result.columns:
        raise ValueError(
            "GSSHA rating curve must contain flow_cfs."
        )

    model_flow = pd.to_numeric(
        result["flow_cfs"],
        errors="coerce"
    ).to_numpy(dtype=float)

    model_aep = np.full(
        len(model_flow),
        np.nan
    )

    # --------------------------------------------------------
    # BELOW REFERENCE MINIMUM
    # --------------------------------------------------------

    below = (
        np.isfinite(model_flow)
        & (model_flow < reference_flow[0])
    )

    model_aep[below] = 1.0

    # --------------------------------------------------------
    # WITHIN REFERENCE RANGE: PCHIP
    # --------------------------------------------------------

    within = (
        np.isfinite(model_flow)
        & (model_flow >= reference_flow[0])
        & (model_flow <= reference_flow[-1])
    )

    model_aep[within] = np.exp(
        probability_interpolator(
            model_flow[within]
        )
    )

    # --------------------------------------------------------
    # ABOVE REFERENCE MAXIMUM
    # LOG-LINEAR EXTRAPOLATION
    # --------------------------------------------------------

    above = (
        np.isfinite(model_flow)
        & (model_flow > reference_flow[-1])
    )

    # Use last two points with distinct probabilities
    last = len(reference_flow) - 1

    previous = last - 1

    while (
        previous > 0
        and np.isclose(
            reference_aep[previous],
            reference_aep[last]
        )
    ):
        previous -= 1

    slope = (
        np.log(reference_aep[last])
        - np.log(reference_aep[previous])
    ) / (
        reference_flow[last]
        - reference_flow[previous]
    )

    if slope >= 0:
        raise ValueError(
            "Cannot extrapolate: final probability slope "
            "must be negative."
        )

    model_aep[above] = np.exp(
        np.log(reference_aep[last])
        + slope * (
            model_flow[above] - reference_flow[last]
        )
    )

    # --------------------------------------------------------
    # ADD PROBABILITY COLUMNS
    # --------------------------------------------------------

    result["EXC_Prob"] = np.clip(
        model_aep,
        0.0,
        1.0
    )

    result["return_period_years"] = (
        1.0 / result["EXC_Prob"]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(f"Watershed: {watershed}")
    print(f"GSSHA simulations: {len(result)}")
    print(
        f"Reference discharge range: "
        f"{reference_flow.min():,.0f}–"
        f"{reference_flow.max():,.0f} cfs"
    )

    print(
        f"Models above reference range: "
        f"{above.sum()}"
    )

    return result

PLOT_SETTINGS = {

    # Figure
    "figsize": (11, 6),
    "dpi": 300,

    # USGS observations
    "usgs_color": "#74B9FF",
    "usgs_size": 22,
    "usgs_edgecolor": "#245A81",
    "usgs_linewidth": 0.8,
    "usgs_zorder": 5,

    # GSSHA model
    "gssha_color": "red",
    "gssha_size": 35,
    "gssha_edgecolor": "black",
    "gssha_zorder": 100,

    # Rating curve
    "curve_color": "#0066CC",
    "curve_width": 1.2,
    "curve_linestyle": "--",
    "curve_zorder": 2,

    # 95% confidence interval
    "ci_color": "#C9DEEF",
    "ci_alpha": 0.75,

    # Breakpoints
    "show_breakpoints": True,
    "show_breakpoint_labels": True,
    "breakpoint_color": "#303030",
    "breakpoint_linewidth": 1.2,
    "breakpoint_alpha": 0.9,
    "breakpoint_fontsize": 9,
    "breakpoint_fontweight": "bold",
    "breakpoint_label_y": 0.97,
    "breakpoint_label_rotation": 0,
    "breakpoint_label_offset": 0.15,

    # Axes
    "x_tick_interval": 2,
    "y_tick_interval": 5000,

    # Fonts
    "title_fontsize": 13,
    "axis_label_fontsize": 12,
    "tick_fontsize": 9,

    # Scientific unit labels
    "x_label": r"Gauge Height (ft (m))",
    "y_label": r"Discharge (ft$^3$/s (m$^3$/s))",

    # Legend
    "legend_location": "upper left",
    "legend_fontsize": 9,
    "legend_names": {
        "Observations": "USGS Observations",
        "Rating Curve": "Rating Curve",
        "Segment Breakpoints": "Breakpoints",
        "Uncertainty": "95% CI",
        "GSSHA": "GSSHA"
    },

    # Grid
    "grid_alpha": 0.25
}


def plot_rating_curve_with_gssha(
    meta,
    gssha_rating_curve,
    title,
    figure_name,
    output_folder="figures",
    min_stage=12,
    max_stage=36,
    min_discharge=0,
    max_discharge=55000,
    settings=None
):
    """
    Plot a fitted Hodson rating curve with USGS observations,
    95% confidence interval, breakpoints, and GSSHA points.

    Stage is plotted in feet with meters underneath.
    Discharge is plotted in ft³/s with m³/s underneath.

    Original input data are not modified.
    """

    # ========================================================
    # SETTINGS
    # ========================================================

    cfg = PLOT_SETTINGS.copy()
    cfg["legend_names"] = PLOT_SETTINGS["legend_names"].copy()

    if settings is not None:
        for key, value in settings.items():
            if key == "legend_names":
                cfg["legend_names"].update(value)
            else:
                cfg[key] = value

    if min_stage >= max_stage:
        raise ValueError(
            "min_stage must be less than max_stage."
        )

    # ========================================================
    # LOAD ORIGINAL USGS DATA
    # ========================================================

    obs = meta["observed_data"].copy()

    obs["stage"] = pd.to_numeric(
        obs["stage"],
        errors="coerce"
    )

    obs["discharge"] = pd.to_numeric(
        obs["discharge"],
        errors="coerce"
    )

    obs = obs.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(subset=["stage", "discharge"])

    # ========================================================
    # LOAD HODSON MODEL PREDICTIONS
    # ========================================================

    unc = meta["uncertainty_bounds"].copy()

    prediction_columns = [
        "stage_ft",
        "lower_cfs",
        "median_cfs",
        "upper_cfs"
    ]

    prediction = unc[prediction_columns].copy()

    for col in prediction_columns:
        prediction[col] = pd.to_numeric(
            prediction[col],
            errors="coerce"
        )

    prediction = (
        prediction
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("stage_ft")
        .drop_duplicates(subset="stage_ft")
    )

    if prediction.empty:
        raise ValueError(
            "No valid Hodson rating curve predictions found."
        )

    h_pred = prediction["stage_ft"].to_numpy()
    q_lower = prediction["lower_cfs"].to_numpy()
    q_median = prediction["median_cfs"].to_numpy()
    q_upper = prediction["upper_cfs"].to_numpy()

    print(
        f"Hodson prediction range: "
        f"{h_pred.min():.2f}–{h_pred.max():.2f} ft"
    )

    # ========================================================
    # LOAD GSSHA DATA
    # ========================================================

    modeled = None

    if gssha_rating_curve is not None:

        modeled = gssha_rating_curve.copy()

        modeled["stage_ft"] = pd.to_numeric(
            modeled["stage_ft"],
            errors="coerce"
        )

        modeled["flow_cfs"] = pd.to_numeric(
            modeled["flow_cfs"],
            errors="coerce"
        )

        modeled = (
            modeled
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["stage_ft", "flow_cfs"])
        )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=cfg["figsize"]
    )

    # ========================================================
    # 95% CONFIDENCE INTERVAL
    # ========================================================

    ax.fill_between(
        h_pred,
        q_lower,
        q_upper,
        color=cfg["ci_color"],
        alpha=cfg["ci_alpha"],
        label="Uncertainty",
        zorder=1
    )

    # ========================================================
    # FITTED HODSON RATING CURVE
    # ========================================================

    ax.plot(
        h_pred,
        q_median,
        color=cfg["curve_color"],
        linewidth=cfg["curve_width"],
        linestyle=cfg["curve_linestyle"],
        label="Rating Curve",
        zorder=cfg["curve_zorder"]
    )

    # ========================================================
    # USGS OBSERVATIONS
    # ========================================================

    ax.scatter(
        obs["stage"],
        obs["discharge"],
        s=cfg["usgs_size"],
        facecolor=cfg["usgs_color"],
        edgecolor=cfg["usgs_edgecolor"],
        linewidth=cfg["usgs_linewidth"],
        label="Observations",
        zorder=cfg["usgs_zorder"]
    )

    # ========================================================
    # SEGMENT BREAKPOINTS
    # ========================================================

    breakpoints = np.asarray(
        meta["equation_parameters"]["hs"][1:],
        dtype=float
    )

    if cfg["show_breakpoints"]:

        for i, bp in enumerate(breakpoints):

            ax.axvline(
                bp,
                color=cfg["breakpoint_color"],
                linestyle="--",
                linewidth=cfg["breakpoint_linewidth"],
                alpha=cfg["breakpoint_alpha"],
                label=(
                    "Segment Breakpoints"
                    if i == 0
                    else "_nolegend_"
                ),
                zorder=2
            )

            if cfg["show_breakpoint_labels"]:

                bp_m = bp * 0.3048

                ax.text(
                    bp + cfg["breakpoint_label_offset"],
                    cfg["breakpoint_label_y"],
                    f"{bp:.2f} ft ({bp_m:.2f} m)",
                    transform=ax.get_xaxis_transform(),
                    rotation=cfg["breakpoint_label_rotation"],
                    fontsize=cfg["breakpoint_fontsize"],
                    fontweight=cfg["breakpoint_fontweight"],
                    color=cfg["breakpoint_color"],
                    ha="left",
                    va="top",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                        pad=2
                    ),
                    zorder=20
                )

    # ========================================================
    # GSSHA MODELED POINTS — ON TOP
    # ========================================================

    if modeled is not None and not modeled.empty:

        ax.scatter(
            modeled["stage_ft"],
            modeled["flow_cfs"],
            s=cfg["gssha_size"],
            color=cfg["gssha_color"],
            edgecolor=cfg["gssha_edgecolor"],
            linewidth=0.5,
            marker="o",
            label="GSSHA",
            zorder=cfg["gssha_zorder"]
        )

    # ========================================================
    # X AXIS — FEET AND METERS
    # ========================================================

    ax.set_xlim(
        min_stage,
        max_stage
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(
            cfg["x_tick_interval"]
        )
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos:
            f"{x:.0f}\n({x * 0.3048:.1f})"
        )
    )

    ax.set_xlabel(
        cfg["x_label"],
        fontsize=cfg["axis_label_fontsize"]
    )

    # ========================================================
    # Y AXIS — FT³/S AND M³/S
    # ========================================================

    if max_discharge is None:

        max_values = [
            float(np.nanmax(q_upper)),
            float(obs["discharge"].max())
        ]

        if modeled is not None and not modeled.empty:
            max_values.append(
                float(modeled["flow_cfs"].max())
            )

        max_discharge = max(max_values) * 1.10

    if min_discharge >= max_discharge:
        raise ValueError(
            "min_discharge must be less than max_discharge."
        )

    ax.set_ylim(
        min_discharge,
        max_discharge
    )

    ax.yaxis.set_major_locator(
        MultipleLocator(
            cfg["y_tick_interval"]
        )
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, pos:
            f"{y:,.0f}\n({y * 0.028316846592:,.0f})"
        )
    )

    ax.set_ylabel(
        cfg["y_label"],
        fontsize=cfg["axis_label_fontsize"]
    )

    ax.tick_params(
        axis="both",
        labelsize=cfg["tick_fontsize"],
        pad=6
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        title,
        fontsize=cfg["title_fontsize"],
        fontweight="bold",
        pad=8
    )

    # ========================================================
    # LEGEND
    # ========================================================

    handles, labels = ax.get_legend_handles_labels()

    legend_order = [
        "Observations",
        "Rating Curve",
        "Segment Breakpoints",
        "Uncertainty",
        "GSSHA"
    ]

    lookup = dict(zip(labels, handles))

    ordered_handles = []
    ordered_labels = []

    for name in legend_order:

        if name in lookup:

            ordered_handles.append(
                lookup[name]
            )

            ordered_labels.append(
                cfg["legend_names"].get(name, name)
            )

    ax.legend(
        ordered_handles,
        ordered_labels,
        fontsize=cfg["legend_fontsize"],
        loc=cfg["legend_location"],
        framealpha=0.9
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        alpha=cfg["grid_alpha"]
    )

    ax.set_axisbelow(True)

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    if figure_name is not None:

        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        figure_path = output_folder / figure_name

        fig.savefig(
            figure_path,
            dpi=cfg["dpi"],
            bbox_inches="tight"
        )

        print(f"Figure saved to: {figure_path.resolve()}")

    # ========================================================
    # SHOW FIGURE
    # ========================================================

    plt.show()

    return fig
