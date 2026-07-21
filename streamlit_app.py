"""
Streamlit interface for the RAG-Powered Document Analyst Crew.
"""

import os
import time
import glob

import streamlit as st

st.set_page_config(
    page_title="Solvane Document Analyst",
    page_icon="🔎",
    layout="centered",
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

HEADER_TOOLTIP = (
    "3-agent CrewAI pipeline (Document Researcher -> Fact Checker -> "
    "Report Writer). Answers strictly from the local knowledge base "
    "(employee handbook, product spec, Q3 financial summary), citing "
    "every claim and flagging anything unsupported."
)

st.markdown(
    """
    <style>
      .solvane-header { display:flex; align-items:center; gap:8px; margin-bottom:0; }
      .solvane-header h2 { margin:0; font-size:1.5rem; font-weight:600; }
      .solvane-info {
          display:inline-flex; align-items:center; justify-content:center;
          width:16px; height:16px; border-radius:50%;
          border:1px solid rgba(150,150,150,0.6);
          font-size:11px; color:rgba(150,150,150,0.9);
          cursor:default; user-select:none;
      }
      .solvane-subtitle { color:rgba(150,150,150,0.9); font-size:0.85rem; margin-top:2px; }
    </style>
    <div class="solvane-header">
      <h2>🔎 Solvane Document Analyst</h2>
      <span class="solvane-info" title="{tooltip}">!</span>
    </div>
    <div class="solvane-subtitle">Ask a question about Solvane Dynamics Inc.</div>
    """.replace("{tooltip}", HEADER_TOOLTIP),
    unsafe_allow_html=True,
)
st.write("")

if "GROQ_API_KEY" not in os.environ:
    with st.expander("⚠️ Groq API key required", expanded=True):
        manual_key = st.text_input(
            "Enter your Groq API key",
            type="password",
            help="Free key: https://console.groq.com/keys",
        )
        if manual_key:
            os.environ["GROQ_API_KEY"] = manual_key
            st.success("Key set for this session.")

DEMO_QUESTIONS = [
    "What is the company's remote work policy and how did Q3 revenue compare to Q2?",
    "What are the core features of Solvane FleetOS, and did Enterprise-tier revenue grow enough in Q3 to justify continued investment in that tier?",
    "What are Solvane's Q4 revenue projections?",
]

with st.form("question_form"):
    preset = st.selectbox(
        "Try a sample question, or choose \"Write my own\" below",
        ["Write my own..."] + DEMO_QUESTIONS,
    )
    default_text = "" if preset == "Write my own..." else preset
    question = st.text_area(
        "Your question",
        value=default_text,
        height=90,
        placeholder="e.g. What is the company's remote work policy?",
    )
    submitted = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

if submitted:
    if "GROQ_API_KEY" not in os.environ:
        st.error("Please provide a Groq API key above first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        from crew import build_crew

        status_box = st.status("Running the crew...", expanded=True)
        try:
            crew = build_crew(question)

            TASK_DONE_LABELS = [
                "🔎 **Document Researcher** — retrieved and drafted an answer from the knowledge base.",
                "✅ **Fact Checker** — independently re-verified each claim.",
                "📝 **Report Writer** — compiled the final report.",
            ]

            def make_callback(label):
                def _on_task_complete(output):
                    status_box.write(label)
                return _on_task_complete

            for i, task in enumerate(crew.tasks):
                task.callback = make_callback(TASK_DONE_LABELS[i])

            status_box.write("🔎 **Document Researcher** — searching the knowledge base...")
            start = time.time()
            result = crew.kickoff()
            elapsed = time.time() - start

            status_box.update(label=f"Done in {elapsed:.0f}s", state="complete", expanded=False)

            st.markdown("---")
            st.markdown(str(result).replace("$", "\\$"))

            os.makedirs("output", exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in question.lower())[:60].strip("_")
            report_path = os.path.join("output", f"report_{slug}.md")
            with open(report_path, "w") as f:
                f.write(str(result))

            st.download_button(
                "⬇️ Download this report (.md)",
                data=str(result),
                file_name=f"{slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as e:
            status_box.update(label="Failed", state="error")
            msg = str(e)
            if "rate_limit" in msg.lower() or "429" in msg:
                st.error(
                    "Groq's free-tier rate limit was hit. Wait about a "
                    "minute and try again — this is a quota limit, not a bug."
                )
            else:
                st.error(f"Something went wrong: {msg}")

existing_reports = sorted(glob.glob(os.path.join("output", "*.md")), key=os.path.getmtime, reverse=True)
if existing_reports:
    st.markdown("---")
    with st.expander(f"📄 Previous reports ({len(existing_reports)})"):
        for path in existing_reports:
            name = os.path.basename(path)
            with open(path) as f:
                content = f.read()
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(name)
            with col2:
                st.download_button("⬇️", data=content, file_name=name, mime="text/markdown", key=path)
            with st.expander("Preview", expanded=False):
                st.markdown(content.replace("$", "\\$"))
