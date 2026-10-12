"""Streamlit dashboard for loan default risk. Run: streamlit run app/dashboard.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditrisk import config, features, model, scoring

st.set_page_config(page_title="Credit Risk Scoring", page_icon=":bank:", layout="wide")

GOOD, BAD, NEUTRAL = "#2a9d8f", "#e76f51", "#264653"
BAND_COLORS = {"A": "#1b998b", "B": "#7fb069", "C": "#f4d35e", "D": "#ee964b", "E": "#d62828"}
OUTCOME = {0: "repaid", 1: "defaulted"}


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_model():
    return model.load()


missing = [p.name for p in (config.SCORED_CSV, config.MODEL_PATH, config.METRICS_JSON,
                            config.IV_CSV, config.INDICATORS_CSV) if not p.exists()]
if missing:
    st.error("Pipeline artefacts not found: " + ", ".join(missing))
    st.code("uv run creditrisk-pipeline", language="bash")
    st.stop()

scored = load_csv(config.SCORED_CSV)
scored["outcome"] = scored["default"].map(OUTCOME)
report = load_json(config.METRICS_JSON)
iv = load_csv(config.IV_CSV)
indicators = load_csv(config.INDICATORS_CSV)
pipe = load_model()
champ_name = report["champion"]
champ = report["models"][champ_name]

# ---------------- sidebar ----------------
st.sidebar.header("Filters")
split = st.sidebar.radio("Rows", ["All applications", "Hold-out test only"])
view = scored if split == "All applications" else scored[scored["split"] == "test"]
purposes = sorted(view["purpose"].unique())
chosen = st.sidebar.multiselect("Loan purpose", purposes, default=purposes)
view = view[view["purpose"].isin(chosen)]
amount_lo, amount_hi = int(scored["credit_amount"].min()), int(scored["credit_amount"].max())
amt = st.sidebar.slider("Credit amount (DM)", amount_lo, amount_hi, (amount_lo, amount_hi), step=100)
view = view[view["credit_amount"].between(*amt)]
if view.empty:
    st.warning("No applications match the filters.")
    st.stop()

# ---------------- header ----------------
st.title("Credit Risk Scoring")
st.caption(f"Probability of default on the Statlog German Credit data. Champion model: "
           f"**{champ_name.replace('_', ' ')}** (selected on cross-validated AUC). "
           "Scores use a 600 @ 50:1 odds, 20 PDO scorecard scale.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Applications", f"{len(view):,}")
c2.metric("Observed default rate", f"{view['default'].mean():.1%}")
c3.metric("Mean predicted PD", f"{view['p_default'].mean():.1%}")
c4.metric("Test ROC-AUC", f"{champ['roc_auc']:.3f}")
c5.metric("Test KS", f"{champ['ks']:.3f}")
c6.metric("Gini", f"{champ['gini']:.3f}")

tabs = st.tabs(["Risk distribution", "Model performance", "Drivers", "Risk indicators",
                "Score an applicant"])

# ---------------- risk distribution ----------------
with tabs[0]:
    left, right = st.columns(2)
    fig = px.histogram(view, x="score", color="outcome", nbins=40, barmode="overlay", opacity=0.75,
                       color_discrete_map={"repaid": GOOD, "defaulted": BAD},
                       title="Score distribution by actual outcome",
                       labels={"score": "scorecard points", "outcome": ""})
    for edge in scoring.BAND_EDGES[1:]:
        fig.add_vline(x=edge, line_dash="dot", line_color="grey")
    left.plotly_chart(fig, width="stretch")

    bands = scoring.band_table(view)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=bands["band"], y=bands["applications"], name="applications",
                          marker_color=[BAND_COLORS[b] for b in bands["band"]]))
    fig2.add_trace(go.Scatter(x=bands["band"], y=bands["default_rate"], name="default rate",
                              yaxis="y2", mode="lines+markers", line=dict(color=NEUTRAL, width=3)))
    fig2.update_layout(title="Applications and default rate per risk band",
                       yaxis=dict(title="applications"),
                       yaxis2=dict(title="default rate", overlaying="y", side="right",
                                   tickformat=".0%", range=[0, 1]),
                       legend=dict(orientation="h", y=1.1))
    right.plotly_chart(fig2, width="stretch")

    show = bands[["band", "action", "applications", "share", "default_rate", "mean_pd",
                  "min_score", "max_score"]].copy()
    show["share"] = show["share"].map("{:.1%}".format)
    show["default_rate"] = show["default_rate"].map("{:.1%}".format)
    show["mean_pd"] = show["mean_pd"].map("{:.1%}".format)
    st.dataframe(show, hide_index=True, width="stretch")

    fig3 = px.box(view, x="purpose", y="p_default", color="outcome", points=False,
                  color_discrete_map={"repaid": GOOD, "defaulted": BAD},
                  title="Predicted PD by loan purpose", labels={"p_default": "PD", "purpose": ""})
    fig3.update_layout(legend_title_text="", yaxis_tickformat=".0%")
    st.plotly_chart(fig3, width="stretch")

# ---------------- model performance ----------------
with tabs[1]:
    st.subheader("Champion vs challenger (hold-out test split)")
    rows = []
    for name, m in report["models"].items():
        rows.append({"model": name.replace("_", " "), "ROC-AUC": m["roc_auc"], "Gini": m["gini"],
                     "KS": m["ks"], "Brier": m["brier"], "CV AUC": f"{m['cv_auc_mean']} ± {m['cv_auc_std']}",
                     "cost threshold": m["threshold"], "expected cost / loan": m["expected_cost"],
                     "precision": m["precision"], "recall": m["recall"],
                     "champion": "yes" if name == champ_name else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Train / test rows: {report['n_train']} / {report['n_test']}. Base default rate "
               f"{report['base_default_rate']:.1%}. The decision threshold minimises the Statlog "
               f"cost matrix (declining a good loan costs 1, approving a bad loan costs 5).")

    a, b, c = st.columns(3)
    fig = go.Figure()
    for name, m in report["models"].items():
        fig.add_trace(go.Scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
                                 name=f"{name.replace('_', ' ')} (AUC {m['roc_auc']:.3f})",
                                 line=dict(width=3 if name == champ_name else 1.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="ROC curve", xaxis_title="false positive rate",
                      yaxis_title="true positive rate", legend=dict(orientation="h", y=-0.25))
    a.plotly_chart(fig, width="stretch")

    ks = champ["ks_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks["threshold"], y=ks["tpr"], name="cum. defaults caught",
                             line=dict(color=BAD)))
    fig.add_trace(go.Scatter(x=ks["threshold"], y=ks["fpr"], name="cum. good declined",
                             line=dict(color=GOOD)))
    fig.update_layout(title=f"KS curve (KS = {champ['ks']:.3f})", xaxis_title="PD threshold",
                      xaxis=dict(autorange="reversed", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.25))
    b.plotly_chart(fig, width="stretch")

    cal = champ["calibration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"],
                             mode="lines+markers", name="model", line=dict(color=NEUTRAL)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title=f"Calibration (Brier {champ['brier']:.3f})",
                      xaxis_title="mean predicted PD", yaxis_title="observed default rate",
                      legend=dict(orientation="h", y=-0.25))
    c.plotly_chart(fig, width="stretch")

    cm = champ["confusion"]
    fig = go.Figure(go.Heatmap(z=cm, x=["predicted repay", "predicted default"],
                               y=["actually repaid", "actually defaulted"], text=cm,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Confusion matrix at threshold {champ['threshold']:.2f}", height=320)
    st.plotly_chart(fig, width="stretch")

# ---------------- drivers ----------------
with tabs[2]:
    left, right = st.columns(2)
    fig = px.bar(iv.sort_values("iv"), x="iv", y="feature", orientation="h", color="strength",
                 title="Information value per feature",
                 category_orders={"strength": ["useless", "weak", "medium", "strong", "suspicious"]},
                 color_discrete_sequence=px.colors.sequential.Teal[2:])
    fig.update_layout(height=600, legend_title_text="", yaxis=dict(dtick=1))
    left.plotly_chart(fig, width="stretch")

    feature = right.selectbox("Weight of evidence by category", iv["feature"].tolist())
    woe = features.woe_table(features.engineer(scored), feature)
    woe["value"] = woe["value"].astype(str)
    fig = px.bar(woe, x="value", y="woe", color="default_rate", color_continuous_scale="RdYlGn_r",
                 title=f"WoE for {feature} (negative = riskier)", labels={"value": "", "woe": "WoE"},
                 hover_data={"n": True, "default_rate": ":.1%"})
    fig.update_layout(height=520, coloraxis_colorbar_title="default rate")
    right.plotly_chart(fig, width="stretch")

    coefs = model.coefficients(pipe)
    if coefs.empty:
        st.info("The champion is a gradient-boosting model; see the logistic challenger for a "
                "linear scorecard view. Information value above is model-agnostic.")
    else:
        top = coefs.head(20).iloc[::-1]
        fig = px.bar(top, x="coefficient", y="feature", orientation="h",
                     color=top["coefficient"] > 0, color_discrete_map={True: BAD, False: GOOD},
                     title="Top 20 standardised logistic coefficients (positive = more risk)")
        fig.update_layout(height=600, showlegend=False, yaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

# ---------------- risk indicators ----------------
with tabs[3]:
    st.subheader("Does the model agree with classic underwriting red flags?")
    st.caption("Each rule marks a subset of applicants. The model agrees when both the observed "
               "default rate and the mean predicted PD for that subset sit on the same side of "
               "the portfolio average.")
    melted = indicators.melt(id_vars="indicator", value_vars=["observed_lift", "predicted_lift"],
                             var_name="kind", value_name="lift")
    melted["kind"] = melted["kind"].str.replace("_lift", "")
    fig = px.bar(melted, x="lift", y="indicator", color="kind", barmode="group", orientation="h",
                 color_discrete_map={"observed": NEUTRAL, "predicted": "#8ecae6"},
                 title="Default-rate lift vs portfolio (1.0 = average)")
    fig.add_vline(x=1, line_dash="dash", line_color="grey")
    fig.update_layout(height=480, legend_title_text="", yaxis=dict(dtick=1, autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    show = indicators.copy()
    for col in ("share", "observed_default_rate", "mean_predicted_pd"):
        show[col] = show[col].map("{:.1%}".format)
    st.dataframe(show, hide_index=True, width="stretch")

# ---------------- score an applicant ----------------
with tabs[4]:
    st.subheader("Score a new application")
    num_cols, cat_cols = features.feature_columns()
    base = scored.iloc[0]
    form = st.columns(4)
    inputs: dict = {}
    inputs["credit_amount"] = form[0].number_input("Credit amount (DM)", 250, 20000, 2500, 50)
    inputs["duration_months"] = form[1].number_input("Duration (months)", 4, 72, 24)
    inputs["installment_rate_pct"] = form[2].number_input("Installment rate (% income)", 1, 4, 3)
    inputs["age"] = form[3].number_input("Age", 18, 80, 35)
    inputs["residence_since_years"] = form[0].number_input("Years at residence", 1, 4, 2)
    inputs["existing_credits"] = form[1].number_input("Existing credits", 1, 4, 1)
    inputs["num_dependents"] = form[2].number_input("Dependents", 1, 2, 1)
    for i, col in enumerate(cat_cols):
        options = sorted(scored[col].unique())
        default_idx = options.index(base[col]) if base[col] in options else 0
        inputs[col] = form[(i + 3) % 4].selectbox(col.replace("_", " "), options, index=default_idx)

    row = features.engineer(pd.DataFrame([inputs]))
    p = float(pipe.predict_proba(row[num_cols + cat_cols])[0, 1])
    score = int(scoring.pd_to_score(p)[0])
    band = scoring.score_to_band(score)
    flags = scoring.flags_for(row.iloc[0], scored)

    r1, r2, r3 = st.columns(3)
    r1.metric("Probability of default", f"{p:.1%}")
    r2.metric("Score", score)
    r3.metric("Band / action", f"{band} · {scoring.BAND_ACTION[band]}")
    st.progress(min(max(p, 0.0), 1.0))
    if flags:
        st.warning("Risk indicators raised: " + "; ".join(flags))
    else:
        st.success("No rule-based risk indicators raised.")
    pct = (scored["score"] < score).mean()
    st.caption(f"This score is higher than {pct:.0%} of the portfolio.")
