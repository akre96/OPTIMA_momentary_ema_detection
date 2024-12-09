from sklearn.metrics import roc_auc_score, average_precision_score
from mhealth_feature_generation import simple_features as sf
import pandas as pd
import numpy as np
from pathlib import Path
from . import config


# FUNCTIONS
def generateHKFeatures(
    user_hk,
    user_id,
    timestamp,
    duration,
    resample="5Min",
):
    data = sf.getDurationAroundTimestamp(user_hk, user_id, timestamp, duration)
    if data.empty:
        return pd.DataFrame()

    active_duration_aggregations = [
        "ActiveEnergyBurned",
        "BasalEnergyBurned",
        "AppleExerciseTime",
        "StepCount",
    ]
    active_duration_agg = [
        sf.aggregateActiveDuration(data, duration_type, resample=resample, qc=True)
        for duration_type in active_duration_aggregations
    ]
    noise = sf.aggregateAudioExposure(data, resample=resample, context="all")
    vital_aggregations = [
        ("HeartRate", (30, 200)),
        ("HeartRateVariabilitySDNN", (0, 1)),
        ("OxygenSaturation", (0.5, 1)),
        ("RespiratoryRate", (0.1, 100)),
    ]
    vital_agg = [
        sf.aggregateVital(
            data,
            vital_type,
            vital_range=vital_range,
            resample=resample,
            context=context,
        )
        for vital_type, vital_range in vital_aggregations
        for context in ["all"]
    ]

    for vital, range in vital_aggregations:
        vital_data = data[data["type"] == vital].copy()
        vital_data = (
            vital_data.loc[
                vital_data["value"].between(range[0], range[1]),
                ["value", "local_start"],
            ]
            .dropna()
            .drop_duplicates()
        )
        vital_data["hours"] = (
            vital_data["local_start"] - vital_data["local_start"].min()
        ) / pd.Timedelta(hours=1)
        vital_data["hours"] = vital_data["hours"].astype(float)
        vital_data = vital_data.sort_values(by="hours")

    hk_metrics = pd.concat(
        [
            noise,
            *active_duration_agg,
            *vital_agg,
        ],
        axis=1,
    )

    # Add QC info
    start, end = sf.calcStartStop(timestamp, duration)
    if end != timestamp:
        print("Warning: end of window does not match end of data")

    hk_metrics["QC_watch_on_percent"] = sf.processWatchOnPercent(
        data, resample=resample, origin=start, end=timestamp
    )
    hk_metrics["QC_watch_on_hours"] = sf.processWatchOnTime(
        data, resample=resample, origin=start
    )
    hk_metrics["QC_duration_days"] = (
        data["local_start"].max() - data["local_start"].min()
    ) / np.timedelta64(1, "D")
    hk_metrics["QC_ndates"] = data["local_start"].dt.date.nunique()

    hk_metrics["user_id"] = user_id
    hk_metrics["survey_start"] = timestamp
    hk_metrics["QC_expected_duration"] = duration
    hk_metrics["QC_expected_duration_minutes"] = pd.to_timedelta(
        duration
    ) / pd.Timedelta(minutes=1)

    return hk_metrics


def gatherUserMetrics(user_id, t_df, windows):
    loader = dl.DataLoader()
    user_hk = loader.loadOPTIMAParticipantData(
        data_folder=Path(config["hk_path"]),
        user_id=user_id,
    )
    metrics = []
    if user_hk.empty:
        return pd.DataFrame()
    for window in windows:
        for ts in t_df.session_start:
            metrics.append(generateHKFeatures(user_hk, user_id, ts, window))
    return pd.concat(metrics)


def bootstrapPerformance(
    prediction_df,
    groupby=[],
    unit="user_id",
    n_boot=500,
    samples_per_group=50,
    random_state=42,
    pred_col="pred_proba",
    true_col="actual",
):

    performance_df = pd.DataFrame(
        columns=[*groupby, "auroc", "auprc", "n"],
        index=range(n_boot),
    )
    use_df = prediction_df.dropna(subset=[true_col, pred_col]).copy()
    use_df[true_col] = use_df[true_col].astype(int)
    for i in range(n_boot):
        bootstrap_sample = use_df.groupby(groupby + [unit]).sample(
            n=samples_per_group, replace=True, random_state=random_state + i
        )

        n = bootstrap_sample.shape[0]
        # Calculate ROC AUC
        auroc = roc_auc_score(bootstrap_sample[true_col], bootstrap_sample[pred_col])

        # Calculate PR AUC
        auprc = average_precision_score(
            bootstrap_sample[true_col], bootstrap_sample[pred_col]
        )

        if len(groupby) > 0:
            performance_df.loc[i] = [
                *bootstrap_sample[groupby].iloc[0],
                auroc,
                auprc,
                n,
            ]
        else:
            performance_df.loc[i] = [auroc, auprc, n]
    performance_df["auroc"] = performance_df["auroc"].astype(float)
    performance_df["auprc"] = performance_df["auprc"].astype(float)
    return performance_df
