import pandas as pd
import matplotlib.pyplot as plt
from ratingcurve.ratings import PowerLawRating
import numpy as np
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import pandas as pd
import matplotlib.pyplot as plt
from ratingcurve.ratings import PowerLawRating
import numpy as np
import os
from pathlib import Path

def define_observed_rating_curve(
    gauge_height,
    discharge,
    segments,
    dates,
    max_discharge,
    uncertainty_bound,
    method="advi",
):
    
    # -----------------------------------
    # Initialize
    # -----------------------------------
        
    meta = {
        "number_segments": segments,
        "final_equation": None,
        "equation_parameters": None,
        "observed_data": None,
        "uncertainty_bounds": None,
        "rating_curve_model_stage_results": None
    }

    fig = None

    # -----------------------------------
    # Clean observed data
    # -----------------------------------
    
    df = pd.DataFrame({
        "stage": gauge_height,
        "discharge": discharge
    })
    
    # Add dates if provided
    if dates is not None:
        df["date"] = dates
    
    # Force stage/discharge to numeric
    df["stage"] = pd.to_numeric(
        df["stage"],
        errors="coerce"
    )
    
    df["discharge"] = pd.to_numeric(
        df["discharge"],
        errors="coerce"
    )
    
    # Remove rows where either stage OR discharge is missing
    df = df.dropna(
        subset=["stage", "discharge"]
    ).copy()
    
    # Sort by stage
    df = df.sort_values("stage").reset_index(drop=True)
    
    # Convert clean data back to arrays for Hodson
    h = df["stage"].to_numpy(dtype=float)
    q = df["discharge"].to_numpy(dtype=float)
    
    print(f"Valid observations: {len(df)}")
    print(f"NaNs in stage: {np.isnan(h).sum()}")
    print(f"NaNs in discharge: {np.isnan(q).sum()}")
    
    meta["observed_data"] = df


    # -----------------------------------
    # Fit Hodson rating curve
    # -----------------------------------
    
    rating = PowerLawRating(segments=segments)

    trace = rating.fit(
        q=q,
        h=h,
        method=method
    )


    # -----------------------------------
    # Get median rating equation
    # -----------------------------------
    
    params = rating.equation()
    meta["equation_parameters"] = params
    
    segments = len(params["b"])

    print("Rating curve equation")
    print("=====================")

    print(
        "ln(q) = a + sum(b[i] * "
        "ln(max(h - hs[i], 0) + ho[i]))"
    )

    print(f"a = {params['a']:.4f}")

    for i in range(segments):
        print(
            f"b[{i}] = {params['b'][i]:.4f}, "
            f"hs[{i}] = {params['hs'][i]:.4f}"
        )

    print()


    # -----------------------------------
    # Build equation string
    # -----------------------------------
    
    terms = []

    for i in range(segments):

        ho = 0 if i == 0 else 1

        if ho == 0:

            terms.append(
                f"{params['b'][i]:.4f} * "
                f"ln(max(h - {params['hs'][i]:.4f}, 0))"
            )

        else:

            terms.append(
                f"{params['b'][i]:.4f} * "
                f"ln(max(h - {params['hs'][i]:.4f}, 0) + 1)"
            )

    eq_str = (
        f"ln(q) = {params['a']:.4f} + "
        + " + ".join(terms)
    )

    print(eq_str)

    meta["final_equation"] = eq_str


    # -----------------------------------
    # Predict discharge at 1-ft stages
    # -----------------------------------
    
    min_stage = params["hs"][0]

    start_stage = np.ceil(min_stage)

    if np.isclose(start_stage, min_stage):
        start_stage += 1


    # Use a large upper stage;
    # max_discharge will actually control cutoff
    stage_pred = np.arange(
        start_stage,
        101,
        1.0
    )


    # Evaluate fitted equation
    ln_q_pred = np.full_like(
        stage_pred,
        params["a"],
        dtype=float
    )

    for i in range(segments):

        ho = 0 if i == 0 else 1

        ln_q_pred += (
            params["b"][i]
            * np.log(
                np.maximum(
                    stage_pred - params["hs"][i],
                    0
                ) + ho
            )
        )

    q_pred = np.exp(ln_q_pred)


    # Limit by maximum discharge
    mask_rating = q_pred <= max_discharge

    stage_results = stage_pred[mask_rating]
    q_results = q_pred[mask_rating]


    model_rating_curve_df = pd.DataFrame({
        "stage_ft": stage_results,
        "discharge_cfs": q_results
    })

    meta["rating_curve_model_stage_results"] = (
        model_rating_curve_df
    )


    # -----------------------------------
    # Posterior predictive uncertainty
    # -----------------------------------
    
    h_pred = np.linspace(
        params["hs"][0] + 0.01,
        100,
        1000
    )

    q_samples = rating.predict_posterior(h_pred)

    tail = (100 - uncertainty_bound) / 2

    q_lower = np.percentile(
        q_samples,
        tail,
        axis=1
    )

    q_median = np.percentile(
        q_samples,
        50,
        axis=1
    )

    q_upper = np.percentile(
        q_samples,
        100 - tail,
        axis=1
    )


    # Store uncertainty WITH stage
    uncertainty_df = pd.DataFrame({
        "stage_ft": h_pred,
        "lower_cfs": q_lower,
        "median_cfs": q_median,
        "upper_cfs": q_upper
    })

    meta["uncertainty_bounds"] = uncertainty_df


    return meta

def plot_rating_curve(
    meta,
    title,
    figure_name=None,
    script_dir=None
):

    # -----------------------------------
    # Get data from meta
    # -----------------------------------

    obs = meta["observed_data"]
    unc = meta["uncertainty_bounds"]
    model_results = meta["rating_curve_model_stage_results"]

    segments = meta["number_segments"]

    # KEEP EVERYTHING IN ORIGINAL UNITS
    # Stage = ft
    # Discharge = cfs
    h = obs["stage"].to_numpy()
    q = obs["discharge"].to_numpy()

    h_pred = unc["stage_ft"].to_numpy()
    q_lower = unc["lower_cfs"].to_numpy()
    q_median = unc["median_cfs"].to_numpy()
    q_upper = unc["upper_cfs"].to_numpy()


    # -----------------------------------
    # Determine modeled curve range
    # -----------------------------------

    max_stage = model_results["stage_ft"].max()

    mask_plot = (
        np.isfinite(q_median) &
        (h_pred <= max_stage)
    )

    curve_stage = h_pred[mask_plot]

    max_discharge_plot = q_median[mask_plot].max()


    # -----------------------------------
    # Create plot
    # -----------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    # -----------------------------------
    # Observations
    # -----------------------------------

    ax.scatter(
        h,
        q,
        s=15,
        color="black",
        label="Observations",
        zorder=3
    )


    # -----------------------------------
    # Median rating curve
    # -----------------------------------

    ax.plot(
        h_pred[mask_plot],
        q_median[mask_plot],
        linewidth=2,
        label=f"{segments} Segment Rating Curve"
    )

    
    params = meta["equation_parameters"]
    
    # hs[0] is the initial stage offset
    # hs[1:] are the actual segment breakpoints
    breakpoints = params["hs"][1:]
    
    for i, bp in enumerate(breakpoints):
    
        ax.axvline(
            bp,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Segment Breakpoints" if i == 0 else None,
            zorder=1
        )
    # --------

    # -----------------------------------
    # Uncertainty
    # -----------------------------------

    ax.fill_between(
        h_pred[mask_plot],
        q_lower[mask_plot],
        q_upper[mask_plot],
        alpha=0.2,
        label="Uncertainty"
    )


    # -----------------------------------
    # X-axis
    # ft with meters underneath
    # -----------------------------------

    min_stage_plot = int(
        np.floor(curve_stage.min())
    )

    max_stage_plot = int(
        np.ceil(curve_stage.max())
    )

    stage_ticks = np.arange(
        min_stage_plot,
        max_stage_plot + 1,
        2
    )

    stage_ticks_metric = stage_ticks * 0.3048

    stage_labels = [
        f"{ft}\n({metric:.1f})"
        for ft, metric in zip(
            stage_ticks,
            stage_ticks_metric
        )
    ]

    ax.set_xticks(stage_ticks)

    ax.set_xticklabels(
        stage_labels,
        fontsize=9
    )

    ax.set_xlim(
        min_stage_plot,
        max_stage_plot
    )


    # -----------------------------------
    # Y-axis
    # cfs with m3/s underneath
    # -----------------------------------

    ax.set_ylim(
        0,
        max_discharge_plot * 1.05
    )

    # Get normal cfs tick locations
    y_ticks = ax.get_yticks()

    # Keep only ticks inside plot
    y_ticks = y_ticks[
        (y_ticks >= 0) &
        (y_ticks <= max_discharge_plot * 1.05)
    ]

    # Convert ONLY the tick-label values
    # cfs -> m3/s
    y_ticks_cms = y_ticks * 0.02831685

    # cfs first line
    # m3/s underneath in parentheses
    y_labels = [
        f"{cfs:,.0f}\n({cms:.0f})"
        for cfs, cms in zip(
            y_ticks,
            y_ticks_cms
        )
    ]

    ax.set_yticks(y_ticks)

    ax.set_yticklabels(
        y_labels,
        fontsize=9
    )


    # -----------------------------------
    # Axis labels
    # -----------------------------------

    ax.set_xlabel(
        "Gauge Height (ft (m))",
        fontsize=11
    )

    ax.set_ylabel(
        r"Discharge (cfs (m$^3$/s))",
        fontsize=11
    )


    # -----------------------------------
    # Title
    # -----------------------------------

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=5
    )


    # -----------------------------------
    # Legend
    # -----------------------------------

    ax.legend(
        fontsize=9
    )


    # -----------------------------------
    # Grid
    # -----------------------------------

    ax.grid(
        alpha=0.3
    )

    ax.set_axisbelow(True)


# -----------------------------------
# Segment breakpoints
# -----------------------------------
# -----------------------------------
# Segment breakpoints
# -----------------------------------

    # Save figure
    # -----------------------------------

    if figure_name is not None:

        if script_dir is None:
            script_dir = os.getcwd()

        figure_folder = os.path.join(
            script_dir,
            "figures"
        )

        os.makedirs(
            figure_folder,
            exist_ok=True
        )

        figure_path = os.path.join(
            figure_folder,
            figure_name
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Figure saved to: {figure_path}"
        )


    # -----------------------------------
    # Show
    # -----------------------------------

    plt.show()

    return fig

def load_watershed_data(
    rating_dir,
    probability_file,
    watershed_files
):

    # -----------------------------------
    # Read probability table
    # -----------------------------------

    probability_df = pd.read_csv(
        probability_file
    )

    probability_df["EXC_Prob"] = pd.to_numeric(
        probability_df["EXC_Prob"],
        errors="coerce"
    )

    # -----------------------------------
    # Initialize dictionary
    # -----------------------------------

    watershed_data = {}

    # -----------------------------------
    # Process each watershed
    # -----------------------------------

    for watershed, config in watershed_files.items():

        rating_file = (
            rating_dir / config["rating_file"]
        )

        probability_column = (
            config["probability_column"]
        )

        # -----------------------------------
        # Load rating curve
        # -----------------------------------

        rating_df = pd.read_csv(
            rating_file
        )

        # Remove exported pandas index column
        rating_df = rating_df.loc[
            :,
            ~rating_df.columns.str.startswith("Unnamed:")
        ].copy()

        # Sort by stage
        rating_df = rating_df.sort_values(
            "stage_ft"
        ).reset_index(drop=True)

        # -----------------------------------
        # Load watershed probability data
        # -----------------------------------

        prob_df = probability_df[
            ["EXC_Prob", probability_column]
        ].copy()

        prob_df = prob_df.rename(
            columns={
                probability_column: "discharge_cfs"
            }
        )

        prob_df = prob_df.dropna(
            subset=["EXC_Prob", "discharge_cfs"]
        )

        # Annual exceedance probability
        prob_df = prob_df[
            (prob_df["EXC_Prob"] > 0)
            & (prob_df["EXC_Prob"] <= 1)
        ].copy()

        # Calculate return period
        prob_df["return_period_years"] = (
            1.0 / prob_df["EXC_Prob"]
        )

        # Sort by return period
        prob_df = prob_df.sort_values(
            "return_period_years"
        ).reset_index(drop=True)

        # -----------------------------------
        # Store in dictionary
        # -----------------------------------

        watershed_data[watershed] = {
            "rating_curve": rating_df,
            "probability": prob_df,
            "rating_file": str(rating_file),
            "probability_column": probability_column
        }

        print(
            f"{watershed}: "
            f"{len(rating_df)} rating points, "
            f"{len(prob_df)} probability points"
        )

    return watershed_data


import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ============================================================
# MERGE RATING CURVE AND FLOOD PROBABILITY
# ============================================================

def merge_rating_probability(
    rating_df,
    probability_df,
    max_return_period=1000
):

    rating = rating_df.copy()
    probability = probability_df.copy()

    # ========================================================
    # PREPARE RATING CURVE
    # ========================================================

    columns = [
        "stage_ft",
        "lower_cfs",
        "median_cfs",
        "upper_cfs"
    ]

    rating[columns] = rating[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    rating = (
        rating.dropna(subset=["stage_ft", "median_cfs"])
        .sort_values("stage_ft")
        .drop_duplicates("stage_ft")
    )

    stage = rating["stage_ft"].to_numpy(dtype=float)
    discharge = rating["median_cfs"].to_numpy(dtype=float)

    if len(stage) < 2:
        raise ValueError("Insufficient rating curve data.")

    if np.any(np.diff(discharge) < 0):
        raise ValueError(
            "Rating curve discharge must be nondecreasing."
        )

    # ========================================================
    # PREPARE PROBABILITY DATA
    # ========================================================

    probability[["discharge_cfs", "EXC_Prob"]] = (
        probability[["discharge_cfs", "EXC_Prob"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    probability = probability.dropna(
        subset=["discharge_cfs", "EXC_Prob"]
    )

    probability = probability[
        (probability["EXC_Prob"] > 0)
        & (probability["EXC_Prob"] <= 1)
    ].copy()

    probability = (
        probability.sort_values("discharge_cfs")
        .drop_duplicates("discharge_cfs")
    )

    prob_q = probability["discharge_cfs"].to_numpy(dtype=float)
    prob_p = probability["EXC_Prob"].to_numpy(dtype=float)

    if len(prob_q) < 2:
        raise ValueError("Insufficient probability data.")

    if np.any(np.diff(prob_p) >= 0):
        raise ValueError(
            "Exceedance probability must decrease with discharge."
        )

    # ========================================================
    # DISCHARGE -> LOG PROBABILITY
    # ========================================================

    probability_interpolator = PchipInterpolator(
        prob_q,
        np.log(prob_p),
        extrapolate=False
    )

    # Log-linear high-flow extrapolation
    slope = (
        np.log(prob_p[-1]) - np.log(prob_p[-2])
    ) / (
        prob_q[-1] - prob_q[-2]
    )

    def predict_probability(q):

        q = np.asarray(q, dtype=float)

        log_p = probability_interpolator(
            np.clip(q, prob_q.min(), prob_q.max())
        )

        log_p = np.where(
            q < prob_q.min(),
            0.0,
            log_p
        )

        log_p = np.where(
            q > prob_q.max(),
            np.log(prob_p[-1])
            + slope * (q - prob_q[-1]),
            log_p
        )

        return np.exp(
            np.clip(log_p, np.log(1e-300), 0.0)
        )

    # ========================================================
    # FIND 1000-YEAR DISCHARGE
    # ========================================================

    target_probability = 1.0 / max_return_period
    target_log_p = np.log(target_probability)

    if target_log_p < np.log(prob_p[-1]):

        # Extrapolate above highest probability discharge
        q_target = (
            prob_q[-1]
            + (
                target_log_p - np.log(prob_p[-1])
            ) / slope
        )

    elif target_log_p > np.log(prob_p[0]):

        raise ValueError(
            "Target return period is below available range."
        )

    else:

        inverse_probability = PchipInterpolator(
            np.log(prob_p[::-1]),
            prob_q[::-1],
            extrapolate=False
        )

        q_target = float(
            inverse_probability(target_log_p)
        )

    # ========================================================
    # INVERT RATING CURVE: DISCHARGE -> STAGE
    # ========================================================

    inverse_data = pd.DataFrame({
        "discharge_cfs": discharge,
        "stage_ft": stage
    })

    inverse_data = (
        inverse_data.sort_values(
            ["discharge_cfs", "stage_ft"]
        )
        .drop_duplicates(
            "discharge_cfs",
            keep="last"
        )
    )

    q_sorted = inverse_data["discharge_cfs"].to_numpy(dtype=float)
    h_sorted = inverse_data["stage_ft"].to_numpy(dtype=float)

    if not (q_sorted.min() <= q_target <= q_sorted.max()):
        raise ValueError(
            f"Target discharge {q_target:,.0f} cfs "
            "is outside the modeled rating curve range."
        )

    inverse_rating = PchipInterpolator(
        q_sorted,
        h_sorted,
        extrapolate=False
    )

    stage_target = float(inverse_rating(q_target))

    # ========================================================
    # CREATE EXACT 1-FT STAGES
    # KEEP CEILING OF 1000-YEAR STAGE
    # ========================================================

    min_stage = int(np.ceil(stage.min()))
    max_stage = int(np.ceil(stage_target))

    if max_stage > stage.max():
        raise ValueError(
            f"Ceiling stage {max_stage} ft exceeds "
            f"rating model maximum {stage.max():.2f} ft."
        )

    stage_1ft = np.arange(
        min_stage,
        max_stage + 1,
        1,
        dtype=int
    )

    rating_1ft = pd.DataFrame({
        "stage_ft": stage_1ft
    })

    # ========================================================
    # INTERPOLATE RATING CURVE AT EACH WHOLE FOOT
    # ========================================================

    for column in [
        "lower_cfs",
        "median_cfs",
        "upper_cfs"
    ]:

        column_data = (
            rating[["stage_ft", column]]
            .dropna()
            .sort_values("stage_ft")
        )

        if len(column_data) < 2:
            raise ValueError(
                f"Insufficient data for {column}."
            )

        interpolator = PchipInterpolator(
            column_data["stage_ft"].to_numpy(dtype=float),
            column_data[column].to_numpy(dtype=float),
            extrapolate=False
        )

        rating_1ft[column] = interpolator(stage_1ft)

    if rating_1ft[
        ["lower_cfs", "median_cfs", "upper_cfs"]
    ].isna().any().any():
        raise ValueError(
            "One or more whole-foot stages are outside "
            "the available uncertainty curve range."
        )

    # ========================================================
    # MERGE PROBABILITY
    # ========================================================

    rating_1ft["EXC_Prob"] = predict_probability(
        rating_1ft["median_cfs"].to_numpy(dtype=float)
    )

    rating_1ft["return_period_years"] = (
        1.0 / rating_1ft["EXC_Prob"]
    )

    # ========================================================
    # METRIC CONVERSIONS
    # ========================================================

    rating_1ft["stage_m"] = (
        rating_1ft["stage_ft"] * 0.3048
    )

    rating_1ft["median_cms"] = (
        rating_1ft["median_cfs"] * 0.028316846592
    )

    # ========================================================
    # ORGANIZE COLUMNS
    # ========================================================

    rating_1ft = rating_1ft[
        [
            "stage_ft",
            "stage_m",
            "median_cfs",
            "median_cms",
            "lower_cfs",
            "upper_cfs",
            "EXC_Prob",
            "return_period_years"
        ]
    ]

    # ========================================================
    # METADATA
    # ========================================================

    info = {
        "target_return_period": max_return_period,
        "q_1000_cfs": float(q_target),
        "stage_1000_ft": stage_target,
        "stage_1000_ceiling_ft": max_stage
    }

    return rating_1ft, probability_interpolator, info


import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ============================================================
# FUNCTION 1: DISCHARGE -> EXCEEDANCE PROBABILITY
# ============================================================

def predict_exceedance_probability(discharge, probability_df):

    probability = probability_df.copy()

    probability["discharge_cfs"] = pd.to_numeric(
        probability["discharge_cfs"], errors="coerce"
    )

    probability["EXC_Prob"] = pd.to_numeric(
        probability["EXC_Prob"], errors="coerce"
    )

    probability = (
        probability.dropna(subset=["discharge_cfs", "EXC_Prob"])
        .loc[
            lambda df:
            (df["EXC_Prob"] > 0) &
            (df["EXC_Prob"] <= 1)
        ]
        .sort_values("discharge_cfs")
        .drop_duplicates("discharge_cfs")
        .reset_index(drop=True)
    )

    prob_q = probability["discharge_cfs"].to_numpy(dtype=float)
    prob_p = probability["EXC_Prob"].to_numpy(dtype=float)

    if len(prob_q) < 2:
        raise ValueError("Insufficient probability points.")

    if np.any(np.diff(prob_p) >= 0):
        raise ValueError(
            "Exceedance probability must decrease with discharge."
        )

    # PCHIP in log-probability space
    interpolator = PchipInterpolator(
        prob_q,
        np.log(prob_p),
        extrapolate=False
    )

    # High-flow log-linear extrapolation
    slope = (
        np.log(prob_p[-1]) - np.log(prob_p[-2])
    ) / (
        prob_q[-1] - prob_q[-2]
    )

    q = np.asarray(discharge, dtype=float)

    log_p = interpolator(
        np.clip(q, prob_q.min(), prob_q.max())
    )

    # Below minimum probability discharge: 100% AEP
    log_p = np.where(
        q < prob_q.min(),
        0.0,
        log_p
    )

    # Above maximum: log-linear extrapolation
    log_p = np.where(
        q > prob_q.max(),
        np.log(prob_p[-1]) + slope * (q - prob_q[-1]),
        log_p
    )

    predicted_probability = np.exp(
        np.clip(log_p, np.log(1e-300), 0.0)
    )

    return predicted_probability


# ============================================================
# FUNCTION 2: RETURN PERIOD -> DISCHARGE
# ============================================================

def discharge_at_return_period(
    probability_df,
    return_period=1000
):

    probability = probability_df.copy()

    probability["discharge_cfs"] = pd.to_numeric(
        probability["discharge_cfs"], errors="coerce"
    )

    probability["EXC_Prob"] = pd.to_numeric(
        probability["EXC_Prob"], errors="coerce"
    )

    probability = (
        probability.dropna(subset=["discharge_cfs", "EXC_Prob"])
        .loc[
            lambda df:
            (df["EXC_Prob"] > 0) &
            (df["EXC_Prob"] <= 1)
        ]
        .sort_values("discharge_cfs")
        .drop_duplicates("discharge_cfs")
        .reset_index(drop=True)
    )

    prob_q = probability["discharge_cfs"].to_numpy(dtype=float)
    prob_p = probability["EXC_Prob"].to_numpy(dtype=float)

    if len(prob_q) < 2:
        raise ValueError("Insufficient probability points.")

    if np.any(np.diff(prob_p) >= 0):
        raise ValueError(
            "Exceedance probability must decrease with discharge."
        )

    target_log_p = np.log(1.0 / return_period)

    # Above highest frequency discharge
    if target_log_p < np.log(prob_p[-1]):

        slope = (
            np.log(prob_p[-1]) - np.log(prob_p[-2])
        ) / (
            prob_q[-1] - prob_q[-2]
        )

        q_target = (
            prob_q[-1] +
            (target_log_p - np.log(prob_p[-1])) / slope
        )

    elif target_log_p > np.log(prob_p[0]):

        raise ValueError(
            "Target return period is below available range."
        )

    else:

        inverse = PchipInterpolator(
            np.log(prob_p[::-1]),
            prob_q[::-1],
            extrapolate=False
        )

        q_target = float(inverse(target_log_p))

    return float(q_target)


# ============================================================
# FUNCTION 3: MERGE RATING CURVE AND PROBABILITY
# EXACT 1-FT STAGES + 1000-YEAR CUTOFF + 1 FT
# ============================================================

def create_stage_probability_table(
    rating_df,
    probability_df,
    max_return_period=1000,
    extra_stage_ft=1
):

    rating = rating_df.copy()

    # --------------------------------------------------------
    # CLEAN RATING DATA
    # --------------------------------------------------------

    columns = [
        "stage_ft",
        "lower_cfs",
        "median_cfs",
        "upper_cfs"
    ]

    rating[columns] = rating[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    rating = (
        rating.dropna(subset=["stage_ft", "median_cfs"])
        .sort_values("stage_ft")
        .drop_duplicates("stage_ft")
        .reset_index(drop=True)
    )

    stage = rating["stage_ft"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # CREATE EXACT 1-FT STAGES
    # --------------------------------------------------------

    stage_1ft = np.arange(
        int(np.ceil(stage.min())),
        int(np.floor(stage.max())) + 1,
        1,
        dtype=int
    )

    result = pd.DataFrame({
        "stage_ft": stage_1ft
    })

    # --------------------------------------------------------
    # INTERPOLATE RATING CURVE AT WHOLE-FOOT STAGES
    # --------------------------------------------------------

    for column in [
        "lower_cfs",
        "median_cfs",
        "upper_cfs"
    ]:

        valid = (
            rating[["stage_ft", column]]
            .dropna()
            .sort_values("stage_ft")
        )

        if len(valid) < 2:
            raise ValueError(
                f"Insufficient data for {column}."
            )

        interpolator = PchipInterpolator(
            valid["stage_ft"].to_numpy(dtype=float),
            valid[column].to_numpy(dtype=float),
            extrapolate=False
        )

        result[column] = interpolator(stage_1ft)

    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    result["EXC_Prob"] = predict_exceedance_probability(
        result["median_cfs"].to_numpy(dtype=float),
        probability_df
    )

    result["return_period_years"] = (
        1.0 / result["EXC_Prob"]
    )

    # --------------------------------------------------------
    # DETERMINE TARGET DISCHARGE
    # --------------------------------------------------------

    q_target = discharge_at_return_period(
        probability_df,
        return_period=max_return_period
    )

    # --------------------------------------------------------
    # FIND FIRST WHOLE-FOOT STAGE REACHING TARGET
    # --------------------------------------------------------

    crossing = np.flatnonzero(
        result["median_cfs"].to_numpy(dtype=float) >= q_target
    )

    target_stage = None
    cutoff_stage = None

    if len(crossing) > 0:

        target_index = int(crossing[0])

        target_stage = int(
            result["stage_ft"].iloc[target_index]
        )

        cutoff_index = min(
            target_index + int(extra_stage_ft),
            len(result) - 1
        )

        cutoff_stage = int(
            result["stage_ft"].iloc[cutoff_index]
        )

        result = result.iloc[
            :cutoff_index + 1
        ].copy()

        if cutoff_stage < target_stage + extra_stage_ft:
            print(
                "WARNING: Insufficient rating-curve range "
                "to include the full extra stage."
            )

    else:

        print(
            f"WARNING: Rating curve does not reach "
            f"the {max_return_period}-year discharge. "
            "All available whole-foot stages retained."
        )

    # --------------------------------------------------------
    # METRIC CONVERSIONS
    # --------------------------------------------------------

    result["stage_m"] = (
        result["stage_ft"] * 0.3048
    )

    result["median_cms"] = (
        result["median_cfs"] * 0.028316846592
    )

    # --------------------------------------------------------
    # ORGANIZE COLUMNS
    # --------------------------------------------------------

    result = result[
        [
            "stage_ft",
            "stage_m",
            "median_cfs",
            "median_cms",
            "lower_cfs",
            "upper_cfs",
            "EXC_Prob",
            "return_period_years"
        ]
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    info = {
        "target_return_period": max_return_period,
        "target_discharge_cfs": q_target,
        "target_stage_ft": target_stage,
        "cutoff_stage_ft": cutoff_stage,
        "extra_stage_ft": extra_stage_ft,
        "reached_target": target_stage is not None
    }

    return result, info
