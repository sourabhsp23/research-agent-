# 🧠 Research Crew — Groq + CrewAI + Streamlit

A *lightweight, multi-agent research assistant* powered by *Groq LLM* and *CrewAI, with a **Streamlit UI* for quick topic exploration.  
It automatically collects facts, summarizes findings, and generates a *polished Markdown report* — ready to download.

---

## 📌 Features

- *Multi-Agent Pipeline*
  - *Researcher* → Gathers key facts and 5 credible sources.
  - *Summarizer* → Condenses findings into crisp, no-fluff bullets.
  - *Report Writer* → Produces a structured, professional Markdown report.

- *External Info Fetching (No API Keys Required)*
  - Wikipedia REST API for summaries.
  - DuckDuckGo HTML parsing for credible links.

- *Groq LLM*
  - Blazing fast generation.
  - Token-conscious prompts (respects Groq’s TPM limits).



- *Markdown Download*
  - Get a clean .md report in one click.

---

## ⚙ Installation



# Install dependencies
pip install streamlit crewai groq requests beautifulsoup4