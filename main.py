import os
import sys
import time
from dotenv import load_dotenv

load_dotenv() 

from crew import build_crew

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

DEMO_QUESTIONS = [
    "What is the company's remote work policy and how did Q3 revenue compare to Q2?",
    "What are the core features of Solvane FleetOS, and did Enterprise-tier revenue grow enough in Q3 to justify continued investment in that tier?",
    "What are Solvane's Q4 revenue projections?",   
]


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:60].strip("_")


def get_questions() -> list[str]:
    """
    Question source, in priority order:
    1. Command-line args:      python main.py "question one" "question two"
    2. Interactive input:      prompted below, blank line to finish
    3. Demo mode:               `python main.py --demo` runs the 3 built-in test questions
    """
    args = sys.argv[1:]

    if args and args[0] == "--demo":
        print("Running in demo mode with the 3 built-in test questions.\n")
        return DEMO_QUESTIONS

    if args:
        return args

    print("Enter your question(s). Press Enter on a blank line when done.")
    print("(Or run `python main.py --demo` to use the 3 built-in test questions.)\n")
    questions = []
    while True:
        q = input(f"Question {len(questions) + 1}: ").strip()
        if not q:
            break
        questions.append(q)

    if not questions:
        print("\nNo questions entered — falling back to demo mode.\n")
        return DEMO_QUESTIONS

    return questions


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit(
            "Set GROQ_API_KEY before running (free key from https://console.groq.com/keys). "
            "This is used for the agent LLM. Embeddings run locally by default and need no key. "
            "If you'd rather use Gemini or OpenAI for the LLM instead, change llm_model in "
            "crew.py's build_crew() and set the matching key."
        )

    questions = get_questions()

    for i, question in enumerate(questions, start=1):
        print(f"\n{'=' * 80}\nQUESTION {i}: {question}\n{'=' * 80}\n")

        crew = build_crew(question)
        result = crew.kickoff()

        report_path = os.path.join(OUTPUT_DIR, f"report_{i}_{slugify(question)}.md")
        with open(report_path, "w") as f:
            f.write(str(result))

        print(f"\nSaved -> {report_path}")


        if i < len(questions):
            print("Pausing 20s before the next question (free-tier rate limit headroom)...")
            time.sleep(20)


if __name__ == "__main__":
    main()
