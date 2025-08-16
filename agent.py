
# pip install streamlit crewai groq requests beautifulsoup4

import streamlit as st
from crewai import Agent, Task, Crew, LLM
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os


# CONFIG 

GROQ_API_KEY = ""  
GROQ_MODEL = "llama-3.1-8b-instant" 

# Guardrail: fail early if key missing
if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
    st.warning("Add your Groq API key at the top of app.py before running.")


# Tiny external fetch tools (no API keys)

def wiki_summary(topic: str, lang: str = "en"):
    """Fetch concise Wikipedia summary via REST API (no key)."""
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            title = data.get("title", topic)
            extract = data.get("extract", "")
            canonical = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            return {"title": title, "summary": extract, "url": canonical}
    except Exception:
        pass
    return {"title": topic, "summary": "", "url": ""}

def ddg_links(query: str, max_results: int = 5):
    """Fetch quick links from DuckDuckGo HTML (no key). Lightweight, best-effort."""
    try:
        url = "https://duckduckgo.com/html/"
        r = requests.post(url, data={"q": query}, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:max_results]:
                href = a.get("href", "")
                text = a.get_text(" ", strip=True)
                if href and text:
                    results.append({"title": text, "url": href})
            return results
    except Exception:
        pass
    return []


# LLM

llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Agents (token-lean roles)

researcher = Agent(
    role="Researcher",
    goal="Collect key facts and 5 credible sources for the topic.",
    backstory="Fast, factual, concise web scout.",
    llm=llm,
    verbose=True 
)

summarizer = Agent(
    role="Summarizer",
    goal="Condense findings into crisp bullets with no fluff.",
    backstory="Clarity-first technical note taker.",
    llm=llm,
    verbose=False
)

report_writer = Agent(
    role="Report Writer",
    goal="Draft a professional Markdown report from bullets + sources.",
    backstory="Writes clean, structured reports with tight language.",
    llm=llm,
    verbose=False
)


# Streamlit UI

st.set_page_config(page_title="CrewAI + Groq Researcher", page_icon="🧠", layout="centered")
st.title("🧠 Conversational Research Crew (Groq + CrewAI)")
st.markdown(
    "This app uses CrewAI and Groq to quickly research topics with a small, efficient crew of agents. "
    "Keep prompts short to respect Groq's 6k TPM constraints."
)


topic = st.text_input("Enter a research topic", placeholder="e.g., Impact of AI on rural healthcare in India")
run_btn = st.button("Run Research Crew")

# Session state for output
if "final_report_md" not in st.session_state:
    st.session_state.final_report_md = ""


# Tasks 

research_task = Task(
    description=(
        "Use the provided external snippets and links to compile facts about the topic: '{topic}'. "
        "Return:\n"
        "1) 5-7 bullet points of key facts (<= 120 words total)\n"
        "2) A short risks/limitations note (<= 40 words)\n"
        "3) A list of 5 sources as markdown list [title](url)\n"
        "Be concise and specific. No repetition."
    ),
    expected_output="Bulleted facts, brief risks, and 5 markdown sources.",
    agent=researcher
)

summary_task = Task(
    description=(
        "Summarize the research facts into at most 8 bullets (<= 110 words total). "
        "Avoid redundancy. Keep only the most decision-useful points."
    ),
    expected_output="<= 8 bullets, <= 110 words total.",
    agent=summarizer
)

report_task = Task(
    description=(
        "Write a final Markdown report from the summary and sources with this exact structure:\n"
        "# {title}\n"
        "Generated on {date}\n\n"
        "## Executive Summary\n"
        "- 5 to 8 bullets (<= 110 words total)\n\n"
        "## Key Insights\n"
        "- 5 bullets, each with 1 line of explanation (<= 120 words total)\n\n"
        "## Risks & Limitations\n"
        "- 3 short bullets\n\n"
        "## References\n"
        "- Exactly 5 markdown links\n"
        "Keep language tight and neutral. Avoid filler."
    ),
    expected_output="Clean Markdown per structure, <= ~350 words.",
    agent=report_writer
)

crew = Crew(
    agents=[researcher, summarizer, report_writer],
    tasks=[research_task, summary_task, report_task],
    verbose=False
)


# Run pipeline

def run_pipeline(user_topic: str):
    # External lightweight lookups
    wiki = wiki_summary(user_topic)
    links = ddg_links(user_topic, max_results=5)

    # Build a compact context block to pass as input (token-aware)
    context_block = [
        f"WIKIPEDIA: {wiki.get('summary','')[:800]}",
        "WIKI_URL: " + (wiki.get("url","") or "N/A"),
        "QUICK_LINKS:\n" + "\n".join([f"- {x['title']}: {x['url']}" for x in links]) if links else "QUICK_LINKS: N/A",
    ]
    context_text = "\n".join(context_block)

    # Inputs provided to the crew once; tasks refer via {topic}
    inputs = {
        "topic": user_topic,
        "title": user_topic.strip().rstrip("."),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "context": context_text
    }

    # Small hint: prepend context to first task dynamically
    research_task.description = (
        f"CONTEXT (external snippets & links):\n{context_text}\n\n" + research_task.description
    )

    # Kickoff
    result = crew.kickoff(inputs=inputs)
    return result

# -----------------------------
# UI actions

if run_btn and topic.strip():
    with st.spinner("Running crew..."):
        try:
            final_md = run_pipeline(topic.strip())
            st.session_state.final_report_md = str(final_md).strip()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.final_report_md:
    st.subheader("📄 Final Report (Markdown)")
    st.code(st.session_state.final_report_md, language="markdown")

    # Download button
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        label="⬇ Download Markdown",
        data=st.session_state.final_report_md.encode("utf-8"),
        file_name=fname,
        mime="text/markdown"
    )

st.markdown("---")
st.caption("Tip: Keep topics specific for best results. This app keeps prompts short to respect Groq's 6k TPM constraints.")



