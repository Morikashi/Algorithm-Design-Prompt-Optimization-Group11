from __future__ import annotations


QA_DATASET = [
    {"input": "Capital of France?", "reference": "Paris"},
    {"input": "2+2?", "reference": "4"},
    {"input": "Largest planet in our solar system?", "reference": "Jupiter"},
    {"input": "Chemical symbol for water?", "reference": "H2O"},
    {"input": "Who wrote Hamlet?", "reference": "Shakespeare"},
]


SUMMARIZATION_DATASET = [
    {
        "input": (
            "A new study reported that regular exercise improves cardiovascular health. "
            "Researchers tracked 5,000 adults over ten years and found lower rates of heart disease. "
            "They recommend at least 150 minutes of moderate activity per week."
        ),
        "reference": "A study tracking 5,000 adults found regular exercise lowers heart disease risk and recommends 150 minutes weekly.",
    },
    {
        "input": (
            "The city council approved a budget to expand public transportation. "
            "The plan adds new bus routes and upgrades stations to improve accessibility. "
            "Officials expect reduced traffic congestion over the next two years."
        ),
        "reference": "The city council approved funding to expand transit with new bus routes and station upgrades to cut congestion.",
    },
]
