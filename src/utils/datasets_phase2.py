from __future__ import annotations

# Phase 2 datasets: >=10 per task, tagged as simple/medium/hard.

QA_PHASE2 = [
    # Simple
    {"difficulty": "simple", "input": "Capital of Italy?", "reference": "Rome"},
    {"difficulty": "simple", "input": "5 * 9?", "reference": "45"},
    {"difficulty": "simple", "input": "Chemical symbol for sodium?", "reference": "Na"},
    {"difficulty": "simple", "input": "Who painted the Mona Lisa?", "reference": "Leonardo da Vinci"},

    # Medium (format-sensitive / short reasoning)
    {"difficulty": "medium", "input": "Is 91 a prime number? Answer yes or no.", "reference": "No"},
    {"difficulty": "medium", "input": "If x=7, compute 3x-2.", "reference": "19"},
    {"difficulty": "medium", "input": "Name the largest ocean. Output only one word.", "reference": "Pacific"},
    {"difficulty": "medium", "input": "Convert 0.75 to a fraction in simplest form.", "reference": "3/4"},

    # Hard (ambiguity / attention to constraints)
    {"difficulty": "hard", "input": "Answer with one word: Which planet is known as the Red Planet?", "reference": "Mars"},
    {"difficulty": "hard", "input": "What is 2^10? Output only the number.", "reference": "1024"},
    {"difficulty": "hard", "input": "How many minutes are in 3.5 hours? Output only the number.", "reference": "210"},
    {"difficulty": "hard", "input": "Is 1 a prime number? Answer yes or no.", "reference": "No"},
]

SUM_PHASE2 = [
    # Simple
    {
        "difficulty": "simple",
        "input": (
            "A company announced a new policy allowing employees to work from home three days a week. "
            "Managers said the change aims to improve work-life balance and reduce commuting costs."
        ),
        "reference": "A company introduced a policy permitting remote work three days weekly to improve work-life balance and cut commuting costs.",
    },
    {
        "difficulty": "simple",
        "input": (
            "A local library expanded weekend hours and added free workshops on resume writing and digital skills. "
            "Officials said the program supports job seekers and students."
        ),
        "reference": "A library expanded weekend hours and launched free resume and digital-skills workshops to support job seekers and students.",
    },

    # Medium (denser facts)
    {
        "difficulty": "medium",
        "input": (
            "A startup released a new app feature that automatically categorizes expenses using receipts. "
            "In early tests, users reduced manual data entry by 60%. The company said accuracy improves with usage, "
            "but warned that unusual receipts may require review."
        ),
        "reference": "A startup added receipt-based expense categorization that cut manual entry by 60% in tests, with accuracy improving over time though unusual receipts may need review.",
    },
    {
        "difficulty": "medium",
        "input": (
            "A city launched a pilot program to replace streetlights with LED fixtures. "
            "Officials estimate energy savings of 35% and lower maintenance costs. "
            "The pilot covers four neighborhoods and will be evaluated after six months."
        ),
        "reference": "A city began a four-neighborhood LED streetlight pilot expected to save 35% energy and reduce maintenance, with results reviewed after six months.",
    },
    {
        "difficulty": "medium",
        "input": (
            "Researchers developed a new water filter that removes microplastics more effectively. "
            "Lab tests showed a 90% reduction, but the team said real-world performance depends on water conditions. "
            "They plan field trials next year."
        ),
        "reference": "Researchers reported a filter that cut microplastics by 90% in lab tests, though real-world results may vary and field trials are planned next year.",
    },

    # Hard (longer + multiple constraints)
    {
        "difficulty": "hard",
        "input": (
            "A hospital introduced an AI tool to assist radiologists in detecting early signs of lung disease. "
            "The system highlights suspicious regions on scans, but clinicians retain full responsibility for diagnosis. "
            "A six-month evaluation found faster review times and fewer missed cases, though administrators emphasized "
            "privacy safeguards and ongoing monitoring for bias."
        ),
        "reference": "A hospital deployed an AI assistant that flags suspicious lung-scan regions to help radiologists, with a six-month review showing faster reads and fewer misses while maintaining clinician responsibility and emphasizing privacy and bias monitoring.",
    },
    {
        "difficulty": "hard",
        "input": (
            "A coastal city approved a $52 million flood mitigation project after repeated storm surges. "
            "Engineers will install movable barriers and upgrade drainage systems over 20 months. "
            "Officials said the plan should reduce damage during severe weather, but residents raised concerns about "
            "construction disruption and long-term maintenance costs."
        ),
        "reference": "A city approved a $52 million, 20-month flood mitigation project with movable barriers and drainage upgrades to reduce storm-surge damage, amid resident concerns about disruption and maintenance costs.",
    },
    {
        "difficulty": "hard",
        "input": (
            "A university revised its admissions policy to expand access for underrepresented students. "
            "The plan adds targeted outreach, increases financial aid, and adjusts evaluation criteria for extracurriculars. "
            "Critics questioned whether standards would change, while administrators said academic expectations remain the same. "
            "The policy will be reviewed after two application cycles."
        ),
        "reference": "A university updated admissions to broaden access via outreach, more aid, and revised extracurricular evaluation, saying standards remain while critics raised concerns; the policy will be reviewed after two cycles.",
    },
    {
        "difficulty": "hard",
        "input": (
            "An airline changed carry-on rules to reduce boarding delays. Passengers may bring one carry-on and one personal item, "
            "and staff will enforce size limits at the gate. The airline expects better on-time performance, but some customers "
            "worried about extra fees for oversized bags."
        ),
        "reference": "An airline tightened carry-on limits and gate enforcement to cut boarding delays and improve on-time performance, though some passengers worry about fees for oversized bags.",
    },
    {
        "difficulty": "hard",
        "input": (
            "A retailer reported revenue up 9% year-over-year due to online growth and improved inventory management. "
            "Executives noted shipping costs remain elevated and may pressure margins. The company plans to expand same-day delivery "
            "to additional cities and invest in warehouse automation."
        ),
        "reference": "A retailer grew revenue 9% from online sales and inventory improvements but warned high shipping costs could hurt margins while expanding same-day delivery and warehouse automation.",
    },
]
