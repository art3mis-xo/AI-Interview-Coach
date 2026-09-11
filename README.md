# AI Interview Coach 🎯

An interactive Streamlit app that simulates adaptive technical interviews for AI/Software Engineering roles.  
The coach generates questions (scenario, conceptual, troubleshooting, design), evaluates answers, and provides feedback with a final report.

## Features
- 📝 **Adaptive Q&A** – Questions adjust in difficulty based on candidate performance.
- 📄 **Job Description Integration** – Paste a JD to generate role‑specific questions using RAG.
- 🔍 **Varied Question Styles** – Scenario, conceptual, troubleshooting, and “how would you…” prompts.
- 📊 **Feedback & Scoring** – Each answer is scored with tips for improvement.
- 📑 **Final Report** – Summary of strengths, weaknesses, and study recommendations.

## Tech Stack
- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – Orchestration
- [Groq LLM](https://groq.com/) – Question generation
- RAG pipeline – Document loading, splitting, vector store, retriever

## Project Structure
InterviewCoach/
├── app.py                  # Streamlit entry point
├── interview_coach.py      # Main InterviewCoach class
├── chains/                 # Chains for interviewer, evaluator, report, question generator
├── rag/                    # RAG utilities (loader, retriever, etc.)
├── data/job_descriptions/  # Sample JD files
├── requirements.txt        # Dependencies
└── README.md               # Project overview

## Installation
```bash
git clone https://github.com/art3mis-xo/AI-Interview-Coach.git
cd AI-Interview-Coach
pip install -r requirements.txt
```

## Usage
Run the app locally:
```bash
streamlit run app.py
```

Open http://localhost:8501 (localhost in Bing) in your browser to start the interview.

## License
MIT License
