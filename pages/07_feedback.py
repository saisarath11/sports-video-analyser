"""
Streamlit Page 7 — Personalised AI Coaching Feedback
"""
import streamlit as st

st.set_page_config(page_title="AI Feedback | AI Sports Analyser", page_icon="🤖", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.feedback-summary {
    background: #12121f; border-left: 4px solid #00c8ff; border-radius: 0 12px 12px 0;
    padding: 18px 22px; font-size: 1.05rem; color: #d0e8ff; margin-bottom: 20px; line-height: 1.6;
}
.engine-badge {
    display: inline-block; background: #1a1a2e; border: 1px solid #7b2ff7;
    border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #c080ff; font-weight: 600;
}
.sugg-box {
    background: #12121f; border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0; font-size: 0.92rem; color: #c0d8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Personalised AI Coaching Feedback")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]
feedback = res.feedback
pids = res.get_player_ids()

if not feedback or not pids:
    st.info("No AI feedback available for this session.")
    st.stop()

selected_pid = st.selectbox(
    "Select Player for Coaching Feedback", pids, format_func=lambda x: f"Player {x}"
)

fb = feedback.get(selected_pid)

if fb:
    badge_label = "🟢 Powered by Gemini LLM" if fb.generated_by == "gemini" else "🔵 Rule-Based Coaching Engine"
    st.markdown(f'<span class="engine-badge">{badge_label}</span>', unsafe_allow_html=True)
    st.write("")

    # Summary box
    st.markdown(f'<div class="feedback-summary"><b>📋 Coaching Summary:</b><br>{fb.summary}</div>', unsafe_allow_html=True)

    # Strengths & Improvements side by side
    col_str, col_imp = st.columns(2)
    with col_str:
        st.subheader("💪 Key Strengths")
        for s in fb.strengths:
            st.success(f"✔ {s}")

    with col_imp:
        st.subheader("🎯 Areas for Improvement")
        for imp in fb.improvements:
            st.warning(f"⚡ {imp}")

    st.divider()

    # Actionable Training Suggestions
    st.subheader("🏋️ Actionable Training Drills & Suggestions")
    for idx, sug in enumerate(fb.training_suggestions, 1):
        st.markdown(f'<div class="sugg-box"><b>Drill {idx}:</b> {sug}</div>', unsafe_allow_html=True)

    st.divider()

    # Recommendations
    st.subheader("💡 Coaching Recommendations")
    for rec in fb.recommendations:
        st.info(f"👉 {rec}")
