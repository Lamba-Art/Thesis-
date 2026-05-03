import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# -----------------------------
# Data creation and preprocessing
# -----------------------------
@st.cache_data
def create_sample_data(n_rows: int = 60, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic employee compensation dataset."""
    rng = np.random.default_rng(seed)

    genders = rng.choice(["Female", "Male"], size=n_rows, p=[0.5, 0.5])
    roles = rng.choice(
        ["Engineer", "Analyst", "Manager", "Designer", "HR"], size=n_rows
    )
    experience = rng.integers(1, 16, size=n_rows)  # 1 to 15 years
    performance = rng.integers(1, 6, size=n_rows)  # 1 to 5 rating

    # Synthetic salary formula with noise (prototype-friendly, explainable).
    base_salary = 30000
    role_adjustment_map = {
        "Engineer": 12000,
        "Analyst": 7000,
        "Manager": 18000,
        "Designer": 8000,
        "HR": 6000,
    }
    role_adjustment = np.array([role_adjustment_map[r] for r in roles])
    noise = rng.normal(0, 8000, size=n_rows)

    salary = (
        base_salary
        + (experience * 2500)
        + (performance * 3500)
        + role_adjustment
        + noise
    )
    salary = np.round(np.clip(salary, 25000, None), 0)

    df = pd.DataFrame(
        {
            "Employee_ID": np.arange(1, n_rows + 1),
            "Gender": genders,
            "Role": roles,
            "Salary": salary,
            "Experience": experience,
            "Performance": performance,
        }
    )
    return df


def normalize_col_name(name: str) -> str:
    """Normalize a column name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def map_required_columns(df: pd.DataFrame) -> dict:
    """Map user-uploaded column names to expected schema names."""
    synonym_groups = {
        "Employee_ID": ["employeeid", "id", "empid", "employee"],
        "Gender": ["gender", "sex"],
        "Salary": ["salary", "pay", "compensation", "wage"],
        "Role": ["role", "job", "title", "jobtitle"],
        "Experience": ["experience", "yearsexperience", "years", "tenure"],
        "Performance": ["performance", "rating", "performancerating", "score"],
    }

    normalized_to_original = {normalize_col_name(c): c for c in df.columns}
    column_map = {}

    for target, synonyms in synonym_groups.items():
        matched_col = None
        for synonym in synonyms:
            if synonym in normalized_to_original:
                matched_col = normalized_to_original[synonym]
                break
        if matched_col:
            column_map[matched_col] = target
    return column_map


def standardize_gender(value: object) -> str:
    """Convert multiple gender formats to Male/Female when possible."""
    if pd.isna(value):
        return np.nan
    token = str(value).strip().lower()
    male_tokens = {"m", "male", "man", "boy"}
    female_tokens = {"f", "female", "woman", "girl"}
    if token in male_tokens:
        return "Male"
    if token in female_tokens:
        return "Female"
    return np.nan


def clean_uploaded_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize schema and clean uploaded dataset."""
    df = raw_df.copy()
    mapped = map_required_columns(df)
    df = df.rename(columns=mapped)

    required_cols = [
        "Employee_ID",
        "Gender",
        "Salary",
        "Role",
        "Experience",
        "Performance",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return pd.DataFrame(), (
            "Uploaded file is missing required columns after auto-detection: "
            + ", ".join(missing)
        )

    # Keep only required columns for this prototype.
    df = df[required_cols].copy()

    # Enforce numeric fields.
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["Experience"] = pd.to_numeric(df["Experience"], errors="coerce")
    df["Performance"] = pd.to_numeric(df["Performance"], errors="coerce")
    df["Employee_ID"] = pd.to_numeric(df["Employee_ID"], errors="coerce")

    # Standardize gender and drop invalid rows.
    df["Gender"] = df["Gender"].apply(standardize_gender)
    df = df.dropna(subset=["Salary", "Gender", "Experience", "Performance", "Employee_ID"])

    # Ensure proper dtypes and clean text role values.
    df["Employee_ID"] = df["Employee_ID"].astype(int)
    df["Role"] = df["Role"].astype(str).str.strip().replace("", "Unknown")
    df["Salary"] = df["Salary"].astype(float)
    df["Experience"] = df["Experience"].astype(float)
    df["Performance"] = df["Performance"].astype(float)

    if df.empty:
        return pd.DataFrame(), "No usable rows remained after cleaning."

    return df, ""


def build_model_features(df: pd.DataFrame, include_role: bool = True) -> pd.DataFrame:
    """Build model features with optional role encoding."""
    X = df[["Experience", "Performance"]].copy()
    if include_role:
        role_dummies = pd.get_dummies(df["Role"], prefix="Role", dtype=float)
        X = pd.concat([X, role_dummies], axis=1)
    return X


@st.cache_resource
def train_salary_model(df: pd.DataFrame, include_role: bool = True) -> tuple[LinearRegression, list[str]]:
    """Train regression model for expected fair salary."""
    X = build_model_features(df, include_role=include_role)
    y = df["Salary"]
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns.tolist()


def predict_fair_salary(
    df: pd.DataFrame, model: LinearRegression, feature_columns: list[str], include_role: bool = True
) -> np.ndarray:
    """Predict fair salary while aligning columns to training schema."""
    X = build_model_features(df, include_role=include_role)
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    return model.predict(X)


def add_fairness_metrics(
    df: pd.DataFrame,
    model: LinearRegression,
    feature_columns: list[str],
    include_role: bool = True,
    salary_column: str = "Salary",
) -> pd.DataFrame:
    """Add predicted salary and inequity flags to a dataframe copy."""
    out = df.copy()
    out["Predicted_Fair_Salary"] = predict_fair_salary(
        out, model, feature_columns, include_role=include_role
    )
    out["Delta_%"] = ((out[salary_column] - out["Predicted_Fair_Salary"]) / out["Predicted_Fair_Salary"]) * 100
    out["Potential_Inequity"] = out["Delta_%"].abs() > 5
    return out


def gender_pay_gap_percent(df: pd.DataFrame, salary_column: str = "Salary") -> float:
    """
    Compute gender pay gap:
    (Average Male - Average Female) / Average Male * 100
    """
    by_gender = df.groupby("Gender")[salary_column].mean()
    male_avg = by_gender.get("Male")
    female_avg = by_gender.get("Female")

    if male_avg is None or female_avg is None or male_avg == 0:
        return 0.0
    return ((male_avg - female_avg) / male_avg) * 100


def build_employee_explanation(row: pd.Series) -> str:
    """Create a human-friendly explanation for one employee."""
    predicted = row["Predicted_Fair_Salary"]
    actual = row["Salary"]
    delta = row["Delta_%"]
    inequity_text = (
        "Potential inequity flagged (>5% difference)."
        if row["Potential_Inequity"]
        else "No significant inequity detected (within 5%)."
    )

    return (
        f"Employee {int(row['Employee_ID'])} is a {row['Role']} with "
        f"{row['Experience']:.0f} years of experience and "
        f"performance score {row['Performance']:.0f}. "
        f"The expected fair salary is ${predicted:,.0f}; actual salary is "
        f"${actual:,.0f} ({delta:+.0f}% vs expected). {inequity_text}"
    )


def answer_chat_question(question: str, df_model: pd.DataFrame) -> str:
    """Very simple chat-style responder for demo questions."""
    q = question.strip().lower()
    if not q:
        return "Please ask a question, for example: 'Why is employee 10 paid this salary?'"

    # Detect employee-focused question.
    if "employee" in q:
        match = re.search(r"employee\s*(\d+)", q)
        if match:
            emp_id = int(match.group(1))
            employee_rows = df_model[df_model["Employee_ID"] == emp_id]
            if employee_rows.empty:
                return f"I couldn't find employee {emp_id}. Please use an ID from the table."
            return build_employee_explanation(employee_rows.iloc[0])
        return "Please include an employee ID, for example: 'Why is employee 10 paid this salary?'"

    # Detect pay-gap question.
    if "gender pay gap" in q or ("pay gap" in q and "gender" in q) or "bias" in q:
        gap = gender_pay_gap_percent(df_model, salary_column="Salary")
        direction = "higher for men" if gap > 0 else "higher for women"
        if abs(gap) < 0.5:
            direction = "effectively balanced"
        return (
            f"Current gender pay gap is {abs(gap):.0f}% ({direction}). "
            "This is based on average salaries by gender in the current dataset."
        )

    return (
        "I can answer questions like: 'Why is employee 10 paid this salary?' "
        "or 'Is there a gender pay gap?'"
    )


def compute_bias_signal(df_model: pd.DataFrame) -> tuple[str, float]:
    """Compare average model delta by gender to detect systematic underpayment."""
    avg_delta_by_gender = df_model.groupby("Gender")["Delta_%"].mean()
    male_delta = float(avg_delta_by_gender.get("Male", 0.0))
    female_delta = float(avg_delta_by_gender.get("Female", 0.0))
    gap_delta = female_delta - male_delta

    if gap_delta < -2:
        msg = "Potential gender bias: females appear more underpaid than males."
    elif gap_delta > 2:
        msg = "Potential gender bias: males appear more underpaid than females."
    else:
        msg = "No strong systematic underpayment signal by gender."
    return msg, gap_delta


def build_recommendations(
    df_model: pd.DataFrame, pay_gap_pct: float, bias_delta_gap: float
) -> list[str]:
    """Generate dynamic recommendations from detected issues."""
    recs = []
    underpaid_count = int((df_model["Salary"] < df_model["Predicted_Fair_Salary"]).sum())
    high_inequity_count = int(df_model["Potential_Inequity"].sum())

    if underpaid_count > 0:
        recs.append(
            f"Adjust salaries for {underpaid_count} underpaid employees toward predicted fair pay."
        )
    if high_inequity_count > 0:
        recs.append(
            f"Prioritize review of {high_inequity_count} employees flagged beyond the 5% fairness threshold."
        )
    if abs(pay_gap_pct) > 3:
        recs.append(
            "Run a formal gender pay equity review and set correction targets by department/role."
        )
    if bias_delta_gap < -2:
        recs.append("Review performance evaluation criteria for possible bias affecting female employees.")
    elif bias_delta_gap > 2:
        recs.append("Review performance evaluation criteria for possible bias affecting male employees.")

    role_gap = (
        df_model.groupby("Role")["Delta_%"].mean().sort_values().head(1)
        if not df_model.empty
        else pd.Series(dtype=float)
    )
    if not role_gap.empty and float(role_gap.iloc[0]) < -3:
        recs.append(
            f"Investigate compensation structure in role '{role_gap.index[0]}' where average delta is most negative."
        )

    recs.append("Audit starting salaries and promotion decisions to prevent compounding inequities over time.")

    return recs


def compute_core_metrics(df_eval: pd.DataFrame, salary_column: str) -> dict:
    """Aggregate metrics used in simulation reporting."""
    return {
        "avg_salary": float(df_eval[salary_column].mean()),
        "pay_gap": float(gender_pay_gap_percent(df_eval, salary_column=salary_column)),
        "flagged": int(df_eval["Potential_Inequity"].sum()),
    }


def apply_budget_allocation(
    df_eval: pd.DataFrame,
    model: LinearRegression,
    feature_columns: list[str],
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Allocate budget to most underpaid employees first."""
    working = df_eval.copy()
    working["Salary_Adjusted"] = working["Salary"]
    queue = working[working["Salary"] < working["Predicted_Fair_Salary"]].copy()
    queue = queue.sort_values("Delta_%", ascending=True)

    remaining = float(budget)
    rows = []

    for _, row in queue.iterrows():
        if remaining <= 0:
            break

        need = float(row["Predicted_Fair_Salary"] - row["Salary_Adjusted"])
        if need <= 0:
            continue

        allocation = min(need, remaining)
        emp_mask = working["Employee_ID"] == row["Employee_ID"]
        old_salary = float(working.loc[emp_mask, "Salary_Adjusted"].iloc[0])
        new_salary = old_salary + allocation
        working.loc[emp_mask, "Salary_Adjusted"] = new_salary
        remaining -= allocation

        rows.append(
            {
                "Employee_ID": int(row["Employee_ID"]),
                "Gender": row["Gender"],
                "Role": row["Role"],
                "Original Salary": old_salary,
                "Adjusted Salary": new_salary,
                "Increase Amount": allocation,
                "Increase %": ((new_salary - old_salary) / old_salary) * 100 if old_salary else 0.0,
                "Reason for Adjustment": f"Underpaid by {abs(float(row['Delta_%'])):.0f}%",
            }
        )

    adjusted_df = working.copy()
    adjusted_df["Salary"] = adjusted_df["Salary_Adjusted"]
    include_role = any(col.startswith("Role_") for col in feature_columns)
    adjusted_eval = add_fairness_metrics(
        adjusted_df,
        model,
        feature_columns,
        include_role=include_role,
        salary_column="Salary",
    )

    changes_df = pd.DataFrame(rows)
    spent = budget - remaining
    return adjusted_eval, changes_df, spent


def compare_strategies(
    df_eval: pd.DataFrame,
    model: LinearRegression,
    feature_columns: list[str],
    budget: float,
) -> pd.DataFrame:
    """Return comparison table for simplified allocation approach."""
    baseline = compute_core_metrics(df_eval, salary_column="Salary")
    out_eval, _, spent = apply_budget_allocation(df_eval, model, feature_columns, budget)
    out_metrics = compute_core_metrics(out_eval, salary_column="Salary")
    return pd.DataFrame(
        [
            {
                "Strategy": "Most underpaid first",
                "Budget Used": spent,
                "Avg Salary (After)": out_metrics["avg_salary"],
                "Pay Gap (Before)": baseline["pay_gap"],
                "Pay Gap (After)": out_metrics["pay_gap"],
                "Flags (Before)": baseline["flagged"],
                "Flags (After)": out_metrics["flagged"],
            }
        ]
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Compensation Fairness Prototype", layout="wide")
st.title("AI-Driven Compensation Fairness Prototype")
st.caption("University demo: explainability, bias detection, and targeted equity correction simulation.")

# -----------------------------
# Upload Data
# -----------------------------
st.markdown("## Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    cleaned_df, upload_error = clean_uploaded_data(raw_df)
    if upload_error:
        st.error(upload_error)
        st.info("Falling back to synthetic sample dataset.")
        df = create_sample_data(n_rows=60)
    else:
        df = cleaned_df
        st.success(f"Uploaded dataset processed successfully. Rows retained: {len(df)}")
else:
    st.info("No file uploaded. Using synthetic sample dataset.")
    df = create_sample_data(n_rows=60)

df_display = df.copy()
for col in ["Employee_ID", "Salary", "Experience", "Performance"]:
    if col in df_display.columns:
        df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(0).astype("Int64")
st.dataframe(df_display.sort_values("Employee_ID"), use_container_width=True, hide_index=True)

# Model and fairness metrics
include_role_in_model = False
model, feature_columns = train_salary_model(df, include_role=include_role_in_model)
df_model = add_fairness_metrics(
    df, model, feature_columns, include_role=include_role_in_model, salary_column="Salary"
)

# -----------------------------
# Dashboard Metrics
# -----------------------------
st.markdown("## Dashboard Metrics")
avg_salary = df_model["Salary"].mean()
gap = gender_pay_gap_percent(df_model, salary_column="Salary")
flagged_count = int(df_model["Potential_Inequity"].sum())
underpaid_count = int((df_model["Salary"] < df_model["Predicted_Fair_Salary"]).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Salary", f"${avg_salary:,.0f}")
col2.metric("Gender Pay Gap", f"{gap:.0f}%")
col3.metric("Flagged Inequities", flagged_count)
col4.metric("Underpaid Employees", underpaid_count)

with st.expander("View employee fairness table", expanded=True):
    display_cols = [
        "Employee_ID",
        "Gender",
        "Role",
        "Experience",
        "Performance",
        "Salary",
        "Predicted_Fair_Salary",
        "Delta_%",
        "Potential_Inequity",
    ]
    fairness_display = df_model[display_cols].copy()
    for col in ["Experience", "Performance", "Salary", "Predicted_Fair_Salary", "Delta_%"]:
        fairness_display[col] = pd.to_numeric(fairness_display[col], errors="coerce").round(0).astype("Int64")
    st.dataframe(fairness_display.sort_values("Employee_ID"), use_container_width=True, hide_index=True)

# -----------------------------
# Fairness Analysis
# -----------------------------
st.markdown("## Fairness Analysis")

selected_employee = st.selectbox(
    "Select Employee ID for explanation",
    options=df_model["Employee_ID"].tolist(),
    index=0,
)
selected_row = df_model[df_model["Employee_ID"] == selected_employee].iloc[0]
st.info(build_employee_explanation(selected_row))

# Gender bias signal based on average model residuals by gender
bias_msg, bias_delta_gap = compute_bias_signal(df_model)
avg_delta_by_gender = (
    df_model.groupby("Gender")["Delta_%"].mean().rename("Average_Delta_%").reset_index()
)
st.write("### Gender Bias Detection")
st.write(bias_msg)
avg_delta_by_gender_display = avg_delta_by_gender.copy()
avg_delta_by_gender_display["Average_Delta_%"] = (
    pd.to_numeric(avg_delta_by_gender_display["Average_Delta_%"], errors="coerce").round(0).astype("Int64")
)
st.dataframe(avg_delta_by_gender_display, use_container_width=True, hide_index=True)

# Root cause tables
st.write("### Root Cause Analysis")
st.caption("Negative Delta_% means actual salary is below model-estimated fair salary.")

role_breakdown = (
    df_model.groupby("Role")
    .agg(
        Employees=("Employee_ID", "count"),
        Avg_Salary=("Salary", "mean"),
        Avg_Predicted_Fair_Salary=("Predicted_Fair_Salary", "mean"),
        Avg_Delta_Percent=("Delta_%", "mean"),
    )
    .reset_index()
    .sort_values("Avg_Delta_Percent")
)
exp_breakdown = (
    df_model.assign(
        Experience_Level=pd.cut(
            df_model["Experience"],
            bins=[-np.inf, 3, 7, 12, np.inf],
            labels=["0-3 yrs", "4-7 yrs", "8-12 yrs", "13+ yrs"],
        )
    )
    .groupby("Experience_Level", observed=True)
    .agg(
        Employees=("Employee_ID", "count"),
        Avg_Salary=("Salary", "mean"),
        Avg_Delta_Percent=("Delta_%", "mean"),
    )
    .reset_index()
)
perf_breakdown = (
    df_model.groupby("Performance")
    .agg(
        Employees=("Employee_ID", "count"),
        Avg_Salary=("Salary", "mean"),
        Avg_Delta_Percent=("Delta_%", "mean"),
    )
    .reset_index()
    .sort_values("Performance")
)

most_underpaid_role = role_breakdown.iloc[0]
most_affected_exp = exp_breakdown.sort_values("Avg_Delta_Percent").iloc[0]
most_underpaid_perf = perf_breakdown.sort_values("Avg_Delta_Percent").iloc[0]

st.write("#### Role-Based Analysis")
role_breakdown_display = role_breakdown.copy()
for col in ["Avg_Salary", "Avg_Predicted_Fair_Salary", "Avg_Delta_Percent"]:
    role_breakdown_display[col] = pd.to_numeric(role_breakdown_display[col], errors="coerce").round(0).astype("Int64")
st.dataframe(role_breakdown_display, use_container_width=True, hide_index=True)
st.caption(
    f"Most underpaid role: {most_underpaid_role['Role']} "
    f"(Avg Delta: {most_underpaid_role['Avg_Delta_Percent']:.0f}%)"
)
st.divider()

st.write("#### Experience-Level Analysis")
exp_breakdown_display = exp_breakdown.copy()
for col in ["Avg_Salary", "Avg_Delta_Percent"]:
    exp_breakdown_display[col] = pd.to_numeric(exp_breakdown_display[col], errors="coerce").round(0).astype("Int64")
st.dataframe(exp_breakdown_display, use_container_width=True, hide_index=True)
st.caption(
    f"Most affected group: {most_affected_exp['Experience_Level']} "
    f"(Avg Delta: {most_affected_exp['Avg_Delta_Percent']:.0f}%)"
)
st.divider()

st.write("#### Performance Analysis")
perf_breakdown_display = perf_breakdown.copy()
for col in ["Avg_Salary", "Avg_Delta_Percent", "Performance"]:
    perf_breakdown_display[col] = pd.to_numeric(perf_breakdown_display[col], errors="coerce").round(0).astype("Int64")
st.dataframe(perf_breakdown_display, use_container_width=True, hide_index=True)
st.caption(
    f"Most underpaid rating: {most_underpaid_perf['Performance']:.0f} "
    f"(Avg Delta: {most_underpaid_perf['Avg_Delta_Percent']:.0f}%)"
)
st.divider()

# -----------------------------
# Simulation (Budget Allocation)
# -----------------------------
st.markdown("## Simulation (Budget Allocation)")
st.caption("Budget is allocated automatically to the most underpaid employees first.")
total_budget = st.number_input(
    "Total Adjustment Budget ($)",
    min_value=0,
    value=50000,
    step=5000,
)
st.info(
    "**How this works:**\n"
    "The system uses a machine learning model to estimate a fair salary for each employee based on "
    "experience, performance, and role.\n\n"
    "When you enter a budget, the system automatically identifies underpaid employees and increases "
    "their salaries toward the predicted fair value. Employees with the largest pay gaps are adjusted "
    "first until the budget is fully used.\n\n"
    "This helps reduce pay inequities and can improve the gender pay gap by prioritizing employees who "
    "are most underpaid."
)

sim_eval, sim_changes, budget_spent = apply_budget_allocation(
    df_model,
    model,
    feature_columns,
    float(total_budget),
)

before_avg = float(df_model["Salary"].mean())
after_avg = float(sim_eval["Salary"].mean())
before_gap = float(gender_pay_gap_percent(df_model, salary_column="Salary"))
after_gap = float(gender_pay_gap_percent(sim_eval, salary_column="Salary"))
before_flagged = int(df_model["Potential_Inequity"].sum())
after_flagged = int(sim_eval["Potential_Inequity"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Cost of Adjustments", f"${budget_spent:,.0f}")
m2.metric("Average Salary", f"${before_avg:,.0f}", delta=f"${(after_avg - before_avg):,.0f}")
m3.metric("Gender Pay Gap", f"{after_gap:.0f}%", delta=f"{(after_gap - before_gap):+.0f}%")
m4.metric("Flagged Inequities", after_flagged, delta=after_flagged - before_flagged)

st.caption(
    f"Cost vs improvement insight: ${budget_spent:,.0f} changes gender pay gap from "
    f"{before_gap:.0f}% to {after_gap:.0f}% and flags from {before_flagged} to {after_flagged}."
)

st.write("### Adjusted Employees (Explainability)")
underpaid_base = df_model[df_model["Salary"] < df_model["Predicted_Fair_Salary"]].copy()
underpaid_display = underpaid_base[
    ["Employee_ID", "Gender", "Role", "Salary"]
].rename(columns={"Salary": "Original Salary"})
underpaid_display["Adjusted Salary"] = underpaid_display["Original Salary"]
underpaid_display["Increase Amount"] = 0.0
underpaid_display["Increase %"] = 0.0
underpaid_display["Adjustment Status"] = "Not Adjusted"

if not sim_changes.empty:
    adjusted_lookup = sim_changes[
        ["Employee_ID", "Adjusted Salary", "Increase Amount", "Increase %"]
    ].copy()
    underpaid_display = underpaid_display.merge(
        adjusted_lookup, on="Employee_ID", how="left", suffixes=("", "_new")
    )
    underpaid_display["Adjusted Salary"] = underpaid_display["Adjusted Salary_new"].fillna(
        underpaid_display["Adjusted Salary"]
    )
    underpaid_display["Increase Amount"] = underpaid_display["Increase Amount_new"].fillna(
        underpaid_display["Increase Amount"]
    )
    underpaid_display["Increase %"] = underpaid_display["Increase %_new"].fillna(
        underpaid_display["Increase %"]
    )
    underpaid_display["Adjustment Status"] = np.where(
        underpaid_display["Employee_ID"].isin(sim_changes["Employee_ID"]),
        "Adjusted",
        "Not Adjusted",
    )
    underpaid_display = underpaid_display.drop(
        columns=["Adjusted Salary_new", "Increase Amount_new", "Increase %_new"]
    )

show_cols = [
    "Employee_ID",
    "Gender",
    "Role",
    "Original Salary",
    "Adjusted Salary",
    "Increase Amount",
    "Increase %",
    "Adjustment Status",
]
for col in ["Original Salary", "Adjusted Salary", "Increase Amount", "Increase %"]:
    underpaid_display[col] = underpaid_display[col].round(0).astype(int)
st.dataframe(
    underpaid_display[show_cols]
    .assign(_status_sort=(underpaid_display["Adjustment Status"] == "Adjusted").astype(int))
    .sort_values(["_status_sort", "Employee_ID"], ascending=[False, True])
    .drop(columns=["_status_sort"]),
    use_container_width=True,
    hide_index=True,
)

st.write("### Scenario Comparison")
comparison_df = compare_strategies(
    df_model,
    model,
    feature_columns,
    float(total_budget),
)
comparison_display = comparison_df.copy()
for col in ["Budget Used", "Avg Salary (After)", "Pay Gap (Before)", "Pay Gap (After)"]:
    comparison_display[col] = comparison_display[col].round(0).astype(int)
st.dataframe(comparison_display, use_container_width=True, hide_index=True)

# -----------------------------
# Insights & Recommendations
# -----------------------------
st.markdown("## Insights & Recommendations")

recommendations = build_recommendations(df_model, gap, bias_delta_gap)
for i, rec in enumerate(recommendations, start=1):
    st.write(f"{i}. {rec}")
