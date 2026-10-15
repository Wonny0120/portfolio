import ollama
from pydantic import BaseModel
from typing import Literal

MODEL = "llama3.2:3b"

CLASSIFIER_SYSTEM = """You assign exactly one task type and exactly one complexity tier.
The two are independent. Decide the task type first, then the tier.

STEP 1 — TASK TYPE.

lookup         - the answer needs information from outside the request: one fact recalled or looked up, or several sources gathered and combined.
judgment       - the answer must be worked out from what is already in the request: calculation, deduction, multi-step inference, or choosing among options and justifying the choice. Nothing is looked up.
classification - the answer is one label from a set the request already lists. No justification needed.
summarization  - restate the user's own text, shorter. Adds no new fact, makes no choice.

Attached or quoted text does NOT make a job summarization — "below", "this thread", "the attached" appear in most tasks. Judge by what the answer must contain:
  "Summarize this article"                        -> summarization
  "Summarize the complaints across these tickets" -> lookup
  "Is the review below positive or negative?"     -> classification

STEP 2 — COMPLEXITY TIER. There are only two.

The test is ONE question: can you start writing the answer immediately, or must
you read and work through something first?

easy — ALL THREE hold:
  - one step only: one fact, one label, one short compression, or one calculation
  - nothing supplied with the request has to be read and applied first
  - two careful people would produce an identical answer

hard — ANY ONE holds:
  - a rule, policy, or set of conditions must be read and applied before answering
  - more than one item must be compared or combined
  - the answer needs a reason attached, not just the result
  - careful people could disagree, or it needs specialist expertise

A short request is not automatically easy. Many one-line requests are hard
because they point at material that has to be read first.

Rate the reasoning required, not whether you happen to know the answer.
A question about very recent information is not hard by itself.
When neither test is clear, choose hard.

PAIRS THAT LOOK ALIKE BUT SPLIT DIFFERENTLY

"Add up these line items and give me the total"          -> easy (just arithmetic)
"Check whether the refund, tax, and credit sum to the
 disputed charge"                                        -> hard (reconcile against a claim)

"Is this a bug, a feature request, or a question?"       -> easy (options are given)
"Is this request within the scope of tier 1 support?"    -> hard (a scope policy must be applied)

"Summarize this paragraph in one sentence"               -> easy (compress one text)
"Summarize this thread for someone who has not read it"  -> hard (decide what matters)

"Pick the cheaper of these two plans"                    -> easy (compare two numbers)
"Which laptop should I buy for data science coursework?" -> hard (fit plus a reason)

"What is the capital of Peru?"                           -> easy (one fact)
"What is the current version of Python?"                 -> easy (still one fact; recent is not hard)

"Is this order over $50?"                                -> easy (read one number)
"Does this order qualify for free shipping?"             -> hard (a policy must be applied)

"Count how many entries in this list are over 100"       -> easy (one pass, one rule stated)
"Which of these customers meets the eligibility rules?"  -> hard (rules must be read first)

"Which of these two dates comes first?"                  -> easy (compare two values)
"Compare the specs and prices of these three GPU models" -> hard (three items, several fields)

Examples:

"Is this message spam or not spam?"                                        -> classification, easy
"Is this review positive, negative, or neutral?"                           -> classification, easy
"How many bytes are in a kilobyte?"                                        -> lookup, easy
"Condense these release notes into five bullets"                           -> summarization, easy
"Solve for x in the equation below"                                        -> judgment, easy
"What is 15 percent of the subtotal shown below?"                          -> judgment, easy
"Sort the numbers below from largest to smallest"                          -> judgment, easy
"Tag this expense as travel, meals, or supplies"                           -> classification, easy

"Route this ticket to billing, technical, or account management"           -> classification, hard
"Mark whether this response violates the tone guidelines"                  -> classification, hard
"Classify this incident as a reportable breach under GDPR"                 -> classification, hard
"Gather the 2023 revenue figures for the top five US airlines"             -> lookup, hard
"What was the exact wording of this clause in the 2019 edition?"           -> lookup, hard
"Turn this incident log into a postmortem timeline"                        -> summarization, hard
"Write an executive summary of this 40-page contract"                      -> summarization, hard
"Debug why this function returns None"                                     -> judgment, hard
"Trace the logic in the code below and say what it returns"                -> judgment, hard
"Which statistical test applies to the dataset described below?"           -> judgment, hard
"Find the flaw in the argument below"                                      -> judgment, hard
"Which vendor best fits our compliance requirements?"                      -> judgment, hard

Reply with JSON only."""


class Tier(BaseModel):
    complexity: Literal["easy", "hard"]
    task: Literal["lookup", "judgment", "classification", "summarization"]


def classify(text: str) -> tuple[str, str, str]:
    """Returns (complexity, task, raw). Falls back to hard on failure."""
    try:
        r = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": text},
            ],
            format=Tier.model_json_schema(),
            options={"temperature": 0, "seed": 42, "num_predict": 64, "num_ctx": 4096},
            keep_alive="30m",
        )
        raw = r["message"]["content"]
        t = Tier.model_validate_json(raw)
        return t.complexity, t.task, raw
    except Exception as e:
        return "hard", "lookup", f"ERROR: {e}"


if __name__ == "__main__":
    for text in ["What is a p-value?"]:
        c, task, _ = classify(text)
        print(f"{c:8} {task:20} | {text}")

    print("\nType a prompt, or 'q' to quit.\n")
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if not prompt:
            continue
        c, task, _ = classify(prompt)
        print(f"{c:8} {task:20} | {prompt}")

    print("bye!")