import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Face Mask Monitoring Dashboard",
    page_icon="😷",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("😷 Real-Time Face Mask Monitoring Dashboard")

st.caption(
    "Analytics dashboard for the Real-Time Face Mask Detection System"
)


LOG_FILE = "detection_log.csv"


# ============================================================
# CHECK LOG FILE
# ============================================================

if not os.path.exists(LOG_FILE):

    st.error(
        "detection_log.csv was not found."
    )

    st.info(
        "Run realtime_detection.py first."
    )

    st.stop()


# ============================================================
# LOAD CSV
# ============================================================

try:

    df = pd.read_csv(
        LOG_FILE,
        header=None,
        on_bad_lines="skip"
    )

except Exception as e:

    st.error(
        f"Unable to read detection_log.csv: {e}"
    )

    st.stop()


# ============================================================
# DETECT CSV FORMAT
# ============================================================

if df.shape[1] >= 6:

    # Newer format with uncertain count

    df = df.iloc[:, :6]

    df.columns = [
        "timestamp",
        "faces_detected",
        "mask_count",
        "no_mask_count",
        "uncertain_count",
        "compliance_percentage"
    ]

elif df.shape[1] == 5:

    # Current format

    df.columns = [
        "timestamp",
        "faces_detected",
        "mask_count",
        "no_mask_count",
        "compliance_percentage"
    ]

    df["uncertain_count"] = 0

else:

    st.error(
        "The detection_log.csv format is invalid."
    )

    st.stop()


# ============================================================
# REMOVE POSSIBLE HEADER ROW
# ============================================================

if len(df) > 0:

    first_value = str(
        df.iloc[0]["timestamp"]
    ).lower()

    if "timestamp" in first_value:

        df = df.iloc[1:].copy()


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "faces_detected",

    "mask_count",

    "no_mask_count",

    "uncertain_count",

    "compliance_percentage"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "timestamp",
        "faces_detected"
    ]
)


# ============================================================
# FILL MISSING VALUES
# ============================================================

for column in numeric_columns:

    df[column] = df[column].fillna(0)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    "timestamp"
)


# ============================================================
# STOP IF EMPTY
# ============================================================

if df.empty:

    st.warning(
        "No valid detection records are available."
    )

    st.stop()


# ============================================================
# CALCULATE TOTALS
# ============================================================

total_face_detections = int(
    df["faces_detected"].sum()
)

total_masks = int(
    df["mask_count"].sum()
)

total_no_masks = int(
    df["no_mask_count"].sum()
)

total_uncertain = int(
    df["uncertain_count"].sum()
)


# ============================================================
# CLASSIFIED DETECTIONS
# ============================================================

total_classified = (

    total_masks
    +
    total_no_masks

)


# ============================================================
# OVERALL COMPLIANCE
# ============================================================

if total_classified > 0:

    overall_compliance = (

        total_masks
        /
        total_classified

    ) * 100

else:

    overall_compliance = 0.0


# ============================================================
# AVERAGES
# ============================================================

average_people_per_frame = (

    df["faces_detected"].mean()

)


average_frame_compliance = (

    df["compliance_percentage"].mean()

)


# ============================================================
# CROWD STATISTICS
# ============================================================

maximum_crowd = int(
    df["faces_detected"].max()
)


minimum_crowd = int(
    df["faces_detected"].min()
)


# ============================================================
# COMPLIANCE STATISTICS
# ============================================================

best_compliance = (

    df["compliance_percentage"].max()

)


lowest_compliance = (

    df["compliance_percentage"].min()

)


# ============================================================
# NUMBER OF RECORDS
# ============================================================

total_records = len(df)


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.subheader(
    "📊 System Overview"
)


# ============================================================
# METRIC ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "👥 Total Face Detections",
        f"{total_face_detections:,}"
    )


with col2:

    st.metric(
        "😷 Mask Detections",
        f"{total_masks:,}"
    )


with col3:

    st.metric(
        "🚨 No-Mask Detections",
        f"{total_no_masks:,}"
    )


with col4:

    st.metric(
        "📈 Overall Compliance",
        f"{overall_compliance:.1f}%"
    )


# ============================================================
# METRIC ROW 2
# ============================================================

col5, col6, col7, col8 = st.columns(4)


with col5:

    st.metric(
        "👥 Maximum Crowd",
        maximum_crowd
    )


with col6:

    st.metric(
        "👤 Avg People / Frame",
        f"{average_people_per_frame:.1f}"
    )


with col7:

    if total_classified > 0:

        no_mask_rate = (

            total_no_masks
            /
            total_classified

        ) * 100

    else:

        no_mask_rate = 0.0


    st.metric(
        "🚨 No-Mask Rate",
        f"{no_mask_rate:.1f}%"
    )


with col8:

    st.metric(
        "📝 Recorded Frames",
        f"{total_records:,}"
    )


# ============================================================
# SEPARATOR
# ============================================================

st.divider()


# ============================================================
# DETECTION DISTRIBUTION
# ============================================================

st.subheader(
    "😷 Detection Distribution"
)


distribution = pd.DataFrame({

    "Category": [

        "Mask",

        "No Mask",

        "Uncertain"

    ],

    "Count": [

        total_masks,

        total_no_masks,

        total_uncertain

    ]

})


col_left, col_right = st.columns(2)


# ============================================================
# PIE CHART
# ============================================================

with col_left:

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    values = [

        total_masks,

        total_no_masks,

        total_uncertain

    ]


    labels = [

        "Mask",

        "No Mask",

        "Uncertain"

    ]


    if sum(values) > 0:

        ax.pie(

            values,

            labels=labels,

            autopct="%1.1f%%",

            startangle=90

        )


        ax.set_title(
            "Mask vs No-Mask"
        )


        st.pyplot(
            fig
        )


    else:

        st.info(
            "No detection statistics available."
        )


    plt.close(fig)


# ============================================================
# BAR CHART
# ============================================================

with col_right:

    st.write(
        "### Detection Counts"
    )


    st.bar_chart(
        distribution.set_index(
            "Category"
        )
    )


# ============================================================
# COMPLIANCE OVER TIME
# ============================================================

st.subheader(
    "📈 Mask Compliance Over Time"
)


compliance_chart = (

    df.set_index(
        "timestamp"
    )[[
        "compliance_percentage"
    ]]

)


st.line_chart(
    compliance_chart
)


# ============================================================
# PEOPLE OVER TIME
# ============================================================

st.subheader(
    "👥 People Detected Over Time"
)


people_chart = (

    df.set_index(
        "timestamp"
    )[[
        "faces_detected"
    ]]

)


st.line_chart(
    people_chart
)


# ============================================================
# MASK VS NO MASK OVER TIME
# ============================================================

st.subheader(
    "😷 Mask vs 🚨 No-Mask Over Time"
)


mask_chart = (

    df.set_index(
        "timestamp"
    )[[
        "mask_count",
        "no_mask_count"
    ]]

)


st.line_chart(
    mask_chart
)


# ============================================================
# COMPLIANCE STATISTICS
# ============================================================

st.subheader(
    "📋 Compliance Statistics"
)


statistics = pd.DataFrame({

    "Metric": [

        "Overall Compliance",

        "Average Frame Compliance",

        "Best Compliance",

        "Lowest Compliance",

        "No-Mask Rate",

        "Maximum Crowd",

        "Minimum Crowd",

        "Average People / Frame"

    ],

    "Value": [

        f"{overall_compliance:.2f}%",

        f"{average_frame_compliance:.2f}%",

        f"{best_compliance:.2f}%",

        f"{lowest_compliance:.2f}%",

        f"{no_mask_rate:.2f}%",

        str(maximum_crowd),

        str(minimum_crowd),

        f"{average_people_per_frame:.2f}"

    ]

})


st.table(
    statistics
)


# ============================================================
# RECENT DETECTION RECORDS
# ============================================================

st.subheader(
    "🕒 Recent Detection Records"
)


recent = (

    df.tail(25)

    .iloc[::-1]

    .copy()

)


# Format timestamp for display

recent["timestamp"] = (

    recent["timestamp"]

    .dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

)


st.dataframe(

    recent,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# NO-MASK EVENTS SUMMARY
# ============================================================

st.subheader(
    "🚨 No-Mask Activity"
)


if total_no_masks > 0:

    st.warning(

        f"⚠️ {total_no_masks:,} no-mask detections "
        f"were recorded."

    )

else:

    st.success(
        "✅ No no-mask detections recorded."
    )


# ============================================================
# DOWNLOAD CLEAN REPORT
# ============================================================

st.subheader(
    "📥 Export Detection Report"
)


export_df = df.copy()


export_df["timestamp"] = (

    export_df["timestamp"]

    .dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

)


csv_data = export_df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(

    label="⬇️ Download CSV Report",

    data=csv_data,

    file_name="face_mask_detection_report.csv",

    mime="text/csv"

)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Dashboard Controls"
)


# ============================================================
# AUTO REFRESH
# ============================================================

auto_refresh = st.sidebar.checkbox(

    "🔄 Auto Refresh",

    value=False

)


if auto_refresh:

    refresh_seconds = st.sidebar.slider(

        "Refresh interval (seconds)",

        min_value=2,

        max_value=30,

        value=5

    )

    import time

    time.sleep(
        refresh_seconds
    )

    st.rerun()


# ============================================================
# SIDEBAR SYSTEM INFORMATION
# ============================================================

st.sidebar.divider()


st.sidebar.subheader(
    "🖥️ System Information"
)


st.sidebar.write(
    "🧠 Model: MobileNetV2"
)

st.sidebar.write(
    "🎯 Classification: Mask / No Mask"
)

st.sidebar.write(
    "👥 Multi-Person Detection"
)

st.sidebar.write(
    "📊 Analytics: Enabled"
)

st.sidebar.write(
    "📝 CSV Logging: Enabled"
)

st.sidebar.write(
    "🎥 Real-Time Detection: Enabled"
)

st.sidebar.write(
    "🔍 Confidence Analysis: Enabled"
)

st.sidebar.write(
    "⏱️ Temporal Smoothing: Enabled"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.sidebar.divider()


st.sidebar.subheader(
    "🧠 Model Performance"
)


st.sidebar.write(
    "Validation Accuracy: ~98.9%"
)

st.sidebar.write(
    "Precision: ~98.9%"
)

st.sidebar.write(
    "Recall: ~98.9%"
)

st.sidebar.write(
    "F1-Score: ~98.9%"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Real-Time Face Mask Detection | "
    "Computer Vision + Deep Learning + "
    "MobileNetV2 + Multi-Person Detection + Analytics"
)