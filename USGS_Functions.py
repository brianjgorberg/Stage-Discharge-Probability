from pathlib import Path
import os
import time
from datetime import date
from io import StringIO

import pandas as pd
import requests

def save_gauge_metadata(gauge_ids, input_file, output_file):

    # Read metadata (keep gauge IDs as strings)
    df = pd.read_csv(
        input_file,
        dtype={"STA_ID": str}
    )

    # Convert requested IDs to strings
    gauge_ids = [str(gauge).strip() for gauge in gauge_ids]

    # Select matching gauges
    selected = df[
        df["STA_ID"].isin(gauge_ids)
    ].copy()

    # Preserve requested gauge order
    selected["STA_ID"] = pd.Categorical(
        selected["STA_ID"],
        categories=gauge_ids,
        ordered=True
    )

    selected = selected.sort_values("STA_ID")

    # Convert back to strings
    selected["STA_ID"] = selected["STA_ID"].astype(str)

    # Create output directory
    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save selected metadata
    selected.to_csv(output_file, index=False)

    # Identify missing gauge IDs
    missing = set(gauge_ids) - set(selected["STA_ID"])

    print(f"Saved {len(selected)} gauges")
    print(f"Output: {output_file}")

    if missing:
        print(f"Gauge IDs not found: {sorted(missing)}")

    return selected
def read_nwis_peak(
    peak_file
):

    # ============================================================
    # READ FILE
    # ============================================================

    with open(peak_file, "r") as f:
        lines = f.readlines()


    # ============================================================
    # FIND DATA TABLE
    # ============================================================

    header_row = next(
        i for i, line in enumerate(lines)
        if line.startswith("agency_cd")
    )


    # ============================================================
    # READ PEAK DATA
    # ============================================================

    peak_data = pd.read_csv(
        peak_file,
        sep="\t",
        skiprows=list(range(header_row)) + [header_row + 1]
    )


    # ============================================================
    # KEEP COLUMNS
    # ============================================================

    peak_data = peak_data[
        [
            "agency_cd",
            "site_no",
            "peak_dt",
            "peak_tm",
            "peak_va",
            "gage_ht"
        ]
    ]


    # ============================================================
    # DATE
    # ============================================================

    peak_data["peak_dt"] = pd.to_datetime(
        peak_data["peak_dt"],
        errors="coerce"
    )


    # ============================================================
    # WATER YEAR
    # October - December = following year
    # ============================================================

    peak_data["water_year"] = (
        peak_data["peak_dt"].dt.year
        + (
            peak_data["peak_dt"].dt.month >= 10
        ).astype(int)
    )


    # ============================================================
    # RETURN
    # ============================================================

    return peak_data

USGS_CONTINUOUS_URL = (
    "https://api.waterdata.usgs.gov/"
    "ogcapi/v1/collections/continuous/items"
)
def _usgs_request(
    url,
    params,
    api_key,
    max_retries=10
):

    headers = {
        "X-Api-Key": api_key
    }

    for attempt in range(max_retries):

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=120
        )

        # Success
        if response.status_code == 200:
            return response

        # Rate limited
        if response.status_code == 429:

            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = 5
            else:
                wait = min(2 ** attempt, 60)

            print(
                f"429 rate limit — "
                f"waiting {wait:.0f} sec..."
            )

            time.sleep(wait)
            continue

        # Any other error
        print(
            f"USGS API error "
            f"{response.status_code}"
        )

        print(response.text[:1000])

        response.raise_for_status()

    raise RuntimeError(
        "USGS request failed after "
        f"{max_retries} retries."
    )


def get_usgs_continuous(
    site_no,
    api_key,
    parameter_codes=("00060", "00065"),
    output_folder=None,
    start_year=None,
    end_year=None,
    limit=50000,
    pause=0.5
):

    site_no = str(site_no).strip()

    monitoring_location_id = (
        f"USGS-{site_no}"
    )

    print()
    print("=" * 70)
    print(f"USGS {site_no}")
    print("=" * 70)

    current_year = pd.Timestamp.now().year

    if start_year is None:
        start_year = 1900

    if end_year is None:
        end_year = current_year

    all_data = []

    # ========================================================
    # PARAMETERS
    # ========================================================

    for parameter_code in parameter_codes:

        print()
        print(
            f"Parameter {parameter_code}"
        )

        parameter_data = []

        # ====================================================
        # YEAR BY YEAR
        # ====================================================

        for year in range(
            start_year,
            end_year + 1
        ):

            print(
                f"    {year}: ",
                end=""
            )

            start_date = (
                f"{year}-01-01T00:00:00Z"
            )

            end_date = (
                f"{year}-12-31T23:59:59Z"
            )

            offset = 0
            year_records = []

            # ================================================
            # PAGINATION
            # ================================================

            while True:

                params = {
                    "monitoring_location_id":
                        monitoring_location_id,

                    "parameter_code":
                        parameter_code,

                    "datetime":
                        f"{start_date}/{end_date}",

                    "limit":
                        limit,

                    "offset":
                        offset,

                    "f":
                        "json"
                }

                response = _usgs_request(
                    USGS_CONTINUOUS_URL,
                    params=params,
                    api_key=api_key
                )

                js = response.json()

                features = js.get(
                    "features",
                    []
                )

                if not features:
                    break

                for feature in features:

                    properties = (
                        feature.get(
                            "properties",
                            {}
                        ).copy()
                    )

                    geometry = feature.get(
                        "geometry"
                    )

                    if (
                        geometry
                        and geometry.get("coordinates")
                    ):

                        coords = geometry[
                            "coordinates"
                        ]

                        if len(coords) >= 2:

                            properties[
                                "longitude"
                            ] = coords[0]

                            properties[
                                "latitude"
                            ] = coords[1]

                    year_records.append(
                        properties
                    )

                n = len(features)

                offset += n

                if n < limit:
                    break

                time.sleep(pause)

            # ================================================
            # YEAR RESULT
            # ================================================

            if year_records:

                year_df = pd.DataFrame(
                    year_records
                )

                parameter_data.append(
                    year_df
                )

                print(
                    f"{len(year_df):,} observations"
                )

            else:

                print("no data")

            time.sleep(pause)

        # ====================================================
        # PARAMETER RESULT
        # ====================================================

        if parameter_data:

            parameter_df = pd.concat(
                parameter_data,
                ignore_index=True
            )

            all_data.append(
                parameter_df
            )

    # ========================================================
    # COMBINE
    # ========================================================

    if not all_data:

        print()
        print("No continuous data found.")

        return pd.DataFrame()

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # ========================================================
    # CLEAN
    # ========================================================

    if "time" in data.columns:

        data["time"] = pd.to_datetime(
            data["time"],
            errors="coerce",
            utc=True
        )

    sort_columns = [
        col
        for col in [
            "parameter_code",
            "time"
        ]
        if col in data.columns
    ]

    if sort_columns:

        data = (
            data
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

    # Remove duplicate observations using scalar identifier fields only
    duplicate_cols = [
        col for col in [
            "time_series_id",
            "time"
        ]
        if col in data.columns
    ]
    
    data = (
        data
        .drop_duplicates(
            subset=duplicate_cols
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SAVE
    # ========================================================

    if output_folder is not None:

        output_folder = Path(
            output_folder
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_folder
            / f"USGS_{site_no}_continuous.csv"
        )

        data.to_csv(
            output_file,
            index=False
        )

        print()
        print(f"Saved: {output_file}")

    print()
    print(
        f"Total observations: "
        f"{len(data):,}"
    )

    return data


def get_start_date_instantaneous(
    usgs_id,
    api_key,
    parameter_codes=("00060", "00065")
):

    usgs_id = str(usgs_id).strip()

    url = (
        "https://api.waterdata.usgs.gov/"
        "ogcapi/v1/collections/"
        "time-series-metadata/items"
    )

    params = {
        "monitoring_location_id": f"USGS-{usgs_id}",
        "limit": 1000,
        "f": "json"
    }

    response = _usgs_request(
        url=url,
        params=params,
        api_key=api_key
    )

    features = response.json().get(
        "features",
        []
    )

    metadata = pd.DataFrame([
        f.get("properties", {})
        for f in features
    ])

    if metadata.empty:
        print(f"USGS {usgs_id}: no records")
        return None


    # -----------------------------------------
    # ONLY CONTINUOUS / INSTANTANEOUS DATA
    # -----------------------------------------

    continuous = metadata[
        (
            metadata["computation_identifier"]
            == "Instantaneous"
        )
        &
        (
            metadata["computation_period_identifier"]
            == "Points"
        )
    ].copy()


    # -----------------------------------------
    # ONLY DISCHARGE / GAGE HEIGHT
    # -----------------------------------------

    continuous = continuous[
        continuous["parameter_code"].isin(
            parameter_codes
        )
    ].copy()


    if continuous.empty:
        print(
            f"USGS {usgs_id}: "
            "no continuous discharge/stage record"
        )
        return None


    # -----------------------------------------
    # Dates
    # -----------------------------------------

    continuous["begin"] = pd.to_datetime(
        continuous["begin"],
        errors="coerce"
    )

    continuous = continuous[
        continuous["begin"].notna()
    ]


    # Show exactly what we're using
    display(
        continuous[
            [
                "parameter_code",
                "parameter_name",
                "begin",
                "end",
                "computation_identifier",
                "computation_period_identifier",
                "primary"
            ]
        ].sort_values("begin")
    )


    start_date = (
        continuous["begin"]
        .min()
        .date()
    )

    print()
    print(
        f"USGS {usgs_id} continuous "
        f"record starts: {start_date}"
    )

    return start_date