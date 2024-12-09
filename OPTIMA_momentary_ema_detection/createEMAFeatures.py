"""
Gather EMA export from PGSQL database and create features for EMA outcomes and prediction
"""

import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
from typing import Tuple
from . import definitions

qid_label_map = definitions["qid_label_map"]
likert_qids = definitions["likert_qids"]
bool_qids = definitions["bool_qids"]


def log_function(x: float, slope: float, intercept: float) -> float:
    return slope * np.log(x) + intercept


def loadEMA(
    ema_path: str,
    timeshift_map: pd.DataFrame,
    qid_label_map: dict,
) -> pd.DataFrame:
    """
    Load EMA data from a CSV file, localize time, and shift by timeshift.
    Add relative days columns for LIFUP1, LIFUP3, Scan1, and Scan2.

    Args:
        ema_path (str): Path to the EMA CSV file.
        participant_data (pd.DataFrame): DataFrame containing participant timing information.
        qid_label_map (dict): Mapping of question IDs to question labels.

    Returns:
        pd.DataFrame: Processed EMA data.
    """
    ema = pd.read_csv(ema_path, parse_dates=["created_at", "response_time"])
    ema["user_id"] = ema["user_id"].astype(int)

    # Merge timeshift data with EMA data
    ema = ema.merge(
        timeshift_map.reset_index().drop_duplicates(),
        on="user_id",
        validate="m:1",
    )

    # Localize time and shift by timeshift
    ema["response_time_loc"] = pd.to_datetime(
        ema["response_time"], format="ISO8601", utc=True
    ).dt.tz_localize(None) + pd.to_timedelta(ema["timeshift"], unit="d")
    ema["question_label"] = ema["q_id"].map(qid_label_map)

    return ema


def getEMABursts(
    ema: pd.DataFrame,
    ema_days: int,
    day_col: str,
    buffer_ahead_ema_days: int,
    user_id_col: str = "user_id",
    day_range: tuple | None = None,
) -> pd.DataFrame:
    """
    Get EMA bursts based on specified criteria.

    Args:
        ema (pd.DataFrame): Processed EMA data.
        ema_days (int): Number of days to consider for EMA bursts before 0.
        day_col (str): Column name representing the day information.
        buffer_ahead_ema_days (int): Number of days to buffer ahead of EMA bursts.
        day_range (tuple | None, optional): Range of days to consider for EMA bursts. Defaults to None.

    Returns:
        pd.DataFrame: EMA bursts data.
    """
    filter_ema = ema.copy()

    if day_range:
        filter_ema = filter_ema[
            filter_ema[day_col].between(*day_range, inclusive="both")
        ]
    else:
        filter_ema = filter_ema[
            filter_ema[day_col].between(
                -1 * ema_days, buffer_ahead_ema_days, inclusive="both"
            )
        ]
    filter_ema["burst_start"] = filter_ema.groupby(
        user_id_col
    ).response_time_loc.transform("min")
    return filter_ema


def getILIADEMABursts(
    ema: pd.DataFrame,
    ema_days: int,
    buffer_ahead_ema_days: int,
    baseline_day_col: str,
    shortterm_day_col: str,
    longterm_ema_lifup1_rel_day_range: tuple,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get ILIADEMA bursts based on specified criteria. Specifically, get baseline, short-term, and long-term EMA bursts.

    Args:
        ema (pd.DataFrame): Processed EMA data.
        ema_days (int): Number of days to consider for EMA bursts.
        buffer_ahead_ema_days (int): Number of days to buffer ahead of EMA bursts.
        baseline_day_col (str): Column name representing the baseline day information.
        shortterm_day_col (str): Column name representing the short-term day information.
        longterm_ema_lifup1_rel_day_range (tuple): Range of days to consider for long-term EMA bursts.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Baseline, short-term, and long-term EMA bursts data.
    """
    # Get baseline EMA bursts
    baseline_ema = getEMABursts(
        ema,
        ema_days,
        baseline_day_col,
        buffer_ahead_ema_days=buffer_ahead_ema_days,
    )

    # Get short-term EMA bursts
    shortterm_ema = getEMABursts(
        ema,
        ema_days,
        shortterm_day_col,
        buffer_ahead_ema_days=buffer_ahead_ema_days,
    )

    # Get long-term EMA bursts
    longterm_ema = getEMABursts(
        ema,
        ema_days,
        baseline_day_col,
        buffer_ahead_ema_days=buffer_ahead_ema_days,
        day_range=longterm_ema_lifup1_rel_day_range,
    )

    return baseline_ema, shortterm_ema, longterm_ema


def qcEMASession(
    ema_df: pd.DataFrame,
    ema_session_duration_range_seconds: tuple,
    user_id_col: str = "user_id",
):
    ema_session_duration = (
        ema_df.groupby([user_id_col, "session_uuid"])
        .response_time_loc.agg(["min", "max"])
        .reset_index()
    )
    n_sessions_initial = ema_session_duration.shape[0]
    ema_session_duration["duration"] = (
        ema_session_duration["max"] - ema_session_duration["min"]
    )
    ema_session_duration["duration_seconds"] = ema_session_duration[
        "duration"
    ].dt.total_seconds()
    # Remove sessions with duration < 10 seconds
    ema_session_duration = ema_session_duration[
        ema_session_duration.duration_seconds.between(
            *ema_session_duration_range_seconds, inclusive="both"
        )
    ]
    print(
        f"Removed {n_sessions_initial - ema_session_duration.shape[0]} sessions with duration outside of range {ema_session_duration_range_seconds}"
    )

    qc_ema_df = ema_df.merge(
        ema_session_duration[["user_id", "session_uuid", "duration_seconds"]],
        on=[user_id_col, "session_uuid"],
        how="inner",
        validate="m:1",
    )
    return qc_ema_df


def qcEMA(
    ema_burst_df: pd.DataFrame,
    ema_session_duration_range_seconds: tuple,
    ema_burst_min_duration_days: int,
    ema_burst_min_sessions: int,
    ema_burst_min_dates: int,
    user_id_col: str = "user_id",
) -> pd.DataFrame:
    """
    Perform quality control checks on EMA bursts data.

    Args:
        ema_burst_df (pd.DataFrame): A single EMA burst of data.
        ema_session_duration_range_seconds (tuple): Range of session durations in seconds.
        ema_burst_min_duration_days (int): Minimum duration in days for EMA bursts.
        ema_burst_min_sessions (int): Minimum number of sessions for EMA bursts.
        ema_burst_min_dates (int): Minimum number of dates for EMA bursts.

    Returns:
        pd.DataFrame: QC-passed EMA bursts data.
    """
    ema_burst_qc = ema_burst_df.copy()

    # Filter sessions to those with duration between 10 seconds and 5 minutes
    ema_session_duration = (
        ema_burst_df.groupby([user_id_col, "session_uuid"])
        .response_time_loc.agg(["min", "max"])
        .reset_index()
    )
    n_sessions_initial = ema_session_duration.shape[0]
    ema_session_duration["duration"] = (
        ema_session_duration["max"] - ema_session_duration["min"]
    )
    ema_session_duration["duration_seconds"] = ema_session_duration[
        "duration"
    ].dt.total_seconds()
    # Remove sessions with duration < 10 seconds
    ema_session_duration = ema_session_duration[
        ema_session_duration.duration_seconds.between(
            *ema_session_duration_range_seconds, inclusive="both"
        )
    ]
    print(
        f"Removed {n_sessions_initial - ema_session_duration.shape[0]} sessions with duration outside of range {ema_session_duration_range_seconds}"
    )

    # remove users with sessions spanning less than ema_burst_min_duration_days
    user_duration = ema_session_duration.groupby(user_id_col).aggregate(
        {"min": "min", "max": "max"}
    )
    user_duration["duration"] = user_duration["max"] - user_duration["min"]
    user_duration["duration_days"] = user_duration["duration"] / pd.Timedelta(days=1)
    user_duration = user_duration[
        user_duration.duration_days >= ema_burst_min_duration_days
    ]
    print(
        f"{ema_session_duration[user_id_col].nunique() - user_duration.shape[0]} users with sessions spanning less than {ema_burst_min_duration_days} days"
    )

    # filter out users with less than ema_burst_min_sessions sessions
    user_sessions = ema_session_duration.groupby(user_id_col).session_uuid.nunique()
    user_sessions = user_sessions[user_sessions >= ema_burst_min_sessions]
    print(
        f"{ema_session_duration[user_id_col].nunique() - user_sessions.shape[0]} users with less than {ema_burst_min_sessions} sessions"
    )

    # filter out users with less than ema_burst_min_dates dates
    ema_session_duration["date"] = ema_session_duration["min"].dt.date
    user_dates = ema_session_duration.groupby(user_id_col)["date"].nunique()
    user_dates = user_dates[user_dates >= ema_burst_min_dates]
    print(
        f"{ema_session_duration[user_id_col].nunique() - user_dates.shape[0]} users with less than {ema_burst_min_dates} dates"
    )

    ema_burst_qc = ema_burst_df[
        ema_burst_df[user_id_col].isin(user_sessions.index)
        & ema_burst_df[user_id_col].isin(user_dates.index)
        & ema_burst_df[user_id_col].isin(user_duration.index)
        & ema_burst_df.session_uuid.isin(ema_session_duration.session_uuid)
    ].copy()
    # Number and % of users removed
    n_users_removed = (
        ema_burst_df[user_id_col].nunique() - ema_burst_qc[user_id_col].nunique()
    )
    n_sessions_removed = n_sessions_initial - ema_burst_qc.session_uuid.nunique()
    percent_users_removed = (
        n_users_removed / ema_burst_df[user_id_col].nunique()
    ) * 100
    percent_sessions_removed = (n_sessions_removed / n_sessions_initial) * 100

    print(
        f"Removed {n_users_removed} total users ({percent_users_removed:.2f}% of total) and {n_sessions_removed} total sessions ({percent_sessions_removed:.2f}% of total)"
    )
    # Map bool answer_ids to true false
    ema_burst_qc.loc[ema_burst_qc.q_id.isin(bool_qids), "answer_id"] = ema_burst_qc.loc[
        ema_burst_qc.q_id.isin(bool_qids), "answer_id"
    ].map({1: 1, 2: 0})

    return ema_burst_qc


def processEMALikertFeatures(single_ema_df: pd.DataFrame):
    """
    Create features from numeric responses to a single EMA question during a burst
    Calculate mean, std, slope, count, duration, and mean_std_interaction

    Args:
        single_ema_df (pd.DataFrame): DataFrame containing likert data for a single EMA.

    Returns:
        pd.Series: Processed likert features.
    """
    if single_ema_df.empty:
        return pd.Series(
            {
                "mean": np.nan,
                "std": np.nan,
                "slope": np.nan,
                "count": 0,
                "duration": np.nan,
                "mean_std_interaction": np.nan,
            }
        )
    single_ema_df = single_ema_df[
        ["answer_id", "response_time_loc", "burst_start"]
    ].dropna()
    single_ema_df["relative_days"] = (
        single_ema_df["response_time_loc"] - single_ema_df["burst_start"]
    ) / pd.Timedelta("1d")
    single_ema_df["answer_id"] = single_ema_df["answer_id"].astype(int)
    if single_ema_df.shape[0] < 3:
        slope = None
        log_slope = None
    else:
        lr = linregress(single_ema_df["relative_days"], single_ema_df["answer_id"])
        slope = round(lr.slope, 10)
        if lr.pvalue > 0.05:
            slope = 0
        # fit a log-linear model
        popt, pcov = curve_fit(
            log_function, single_ema_df["relative_days"], single_ema_df["answer_id"]
        )
        log_slope = popt[0]

    mean = single_ema_df["answer_id"].mean()
    std = single_ema_df["answer_id"].std()
    mean_std_interaction = mean / std
    count = single_ema_df["answer_id"].count()
    duration = (
        single_ema_df["relative_days"].max() - single_ema_df["relative_days"].min()
    )
    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "slope": slope,
            "log_slope": log_slope,
            "count": count,
            "duration": duration,
            "coef_of_var": mean_std_interaction,
        }
    )


def aggregateEMALikertBool(ema_burst_df: pd.DataFrame, user_id_col: str):
    """
    Aggregate likert and boolean features for EMA bursts.
    Args:
        ema_burst_df (pd.DataFrame): EMA bursts data.

    Returns:
        pd.DataFrame: Aggregated likert and boolean features.
    """
    likert_agg = (
        ema_burst_df[ema_burst_df.q_id.isin(likert_qids + bool_qids)]
        .groupby([user_id_col, "q_id"])
        .apply(processEMALikertFeatures, include_groups=False)
        .reset_index()
    )
    likert_agg["q_label"] = likert_agg.q_id.map(qid_label_map)
    likert_agg = likert_agg.pivot_table(
        index=user_id_col,
        columns="q_label",
        values=[
            "mean",
            "std",
            "slope",
            "log_slope",
            "count",
            "duration",
            "coef_of_var",
        ],
    )

    # flatten multiindex columns
    likert_agg.columns = [col[1] + "-" + col[0] for col in likert_agg.columns]
    return likert_agg


def aggregateEMAPosNegAffect(
    ema_burst_df: pd.DataFrame,
    pos_affect_qids: list,
    neg_affect_qids: list,
    user_id_col: str,
):
    """
    Calculates average positive affect, negative affect, and the ratio of positive to negative affect for EMA bursts.

    Args:
        ema_burst_df (pd.DataFrame): EMA bursts data.
        pos_affect_qids (list): List of question IDs for positive affect.
        neg_affect_qids (list): List of question IDs for negative affect.

    Returns:
        pd.DataFrame: Aggregated positive and negative affect features.
    """
    ema_burst_subset = ema_burst_df.loc[
        ema_burst_df.q_id.isin(pos_affect_qids + neg_affect_qids),
        [user_id_col, "session_uuid", "q_id", "answer_id"],
    ].copy()
    ema_burst_piv = ema_burst_subset.pivot(
        index=[user_id_col, "session_uuid"], columns="q_id", values="answer_id"
    ).dropna()
    ema_burst_piv["PA"] = ema_burst_piv[pos_affect_qids].mean(axis=1)
    ema_burst_piv["NA"] = ema_burst_piv[neg_affect_qids].mean(axis=1)
    ema_burst_piv["PA:NA"] = ema_burst_piv["PA"] / ema_burst_piv["NA"]
    ema_burst_piv = ema_burst_piv[["PA", "NA", "PA:NA"]].reset_index()
    ema_burst_piv = ema_burst_piv.groupby(user_id_col)[["PA", "NA", "PA:NA"]].mean()
    return ema_burst_piv


def label_company_context(row):
    if row[17] == 8:
        return "company_alone"
    elif row[18] == 4:
        return "company_neutral"
    elif row[18] < 4:
        return "company_bad"
    else:
        return "company_good"


def label_activity_context(row):
    if row[20] >= 4:
        return definitions["activity_label_map"][row[19]] + "-enj"
    elif row[20] < 4:
        return definitions["activity_label_map"][row[19]] + "-nenj"
    else:
        return None


def handleMultipleContextResponses(ema_burst_df, context_qids):
    context_ema_burst = ema_burst_df[ema_burst_df.q_id.isin(context_qids)].copy()
    # if multiple responses for a context question 19 or 20, create new row for each response, keep other columns the same

    sessions_to_split = context_ema_burst.groupby("session_uuid").q_id.value_counts()
    sessions_to_split = sessions_to_split[sessions_to_split > 1]

    for (session_uuid, q_id), row in pd.DataFrame(sessions_to_split).iterrows():
        og_session = context_ema_burst[
            context_ema_burst.session_uuid == session_uuid
        ].copy()
        repeated_answer_rows = og_session[og_session.q_id == q_id].copy()
        # Remove og session from context_ema_burst
        context_ema_burst = context_ema_burst[
            context_ema_burst.session_uuid != session_uuid
        ]

        trimmed_session = og_session[
            (og_session.q_id != q_id)
            | (
                (og_session.q_id == q_id)
                & (og_session.answer_id == repeated_answer_rows.answer_id.iloc[0])
            )
        ].copy()
        sessions_to_append = [trimmed_session]
        for j, val in enumerate(og_session[og_session.q_id == q_id].answer_id):
            if j == 0:
                continue
            new_session = trimmed_session.copy()
            new_session.loc[new_session.q_id == q_id, "answer_id"] = val
            new_session["session_uuid"] = new_session["session_uuid"] + f"-{j}"
            sessions_to_append.append(new_session)
        context_ema_burst = pd.concat([context_ema_burst] + sessions_to_append)
    return context_ema_burst


def aggregateEMAContext(ema_burst_df: pd.DataFrame, user_id_col: str):

    activity_context_df = handleMultipleContextResponses(ema_burst_df, [19, 20])
    activity_context_piv = activity_context_df.pivot_table(
        index=["session_uuid", user_id_col],
        columns="q_id",
        values="answer_id",
    ).dropna()

    company_context_df = handleMultipleContextResponses(ema_burst_df, [17, 18])
    company_context_piv = company_context_df.pivot_table(
        index=["session_uuid", user_id_col],
        columns="q_id",
        values="answer_id",
    ).dropna()

    activity_context_piv["activity"] = activity_context_piv.apply(
        label_activity_context, axis=1
    )
    company_context_piv["company"] = company_context_piv.apply(
        label_company_context, axis=1
    )

    company_context_time = (
        (
            company_context_piv.reset_index()
            .groupby([user_id_col, "company"])
            .session_uuid.count()
            .reset_index()
            .pivot_table(index=[user_id_col], columns="company", values="session_uuid")
            .apply(lambda x: x / x.sum(), axis=1)
        )
        .dropna(how="all")
        .fillna(0)
    )

    activity_context_time = (
        (
            activity_context_piv.reset_index()
            .groupby([user_id_col, "activity"])
            .session_uuid.count()
            .reset_index()
            .pivot_table(index=[user_id_col], columns="activity", values="session_uuid")
            .apply(lambda x: x / x.sum(), axis=1)
        )
        .dropna(how="all")
        .fillna(0)
    )

    context_time = pd.concat([company_context_time, activity_context_time], axis=1)

    return context_time
