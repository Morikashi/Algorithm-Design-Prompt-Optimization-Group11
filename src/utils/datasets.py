from __future__ import annotations

# Phase-1 datasets: reference-based, small enough to run fast, large enough to compare methods.

QA_DATASET = [
    # Geography / facts
    {"input": "Capital of France?", "reference": "Paris"},
    {"input": "Capital of Japan?", "reference": "Tokyo"},
    {"input": "Largest planet in our solar system?", "reference": "Jupiter"},
    {"input": "Which ocean is the largest?", "reference": "Pacific"},
    {"input": "Mount Everest is in which mountain range?", "reference": "Himalayas"},

    # Science
    {"input": "Chemical symbol for water?", "reference": "H2O"},
    {"input": "Chemical symbol for gold?", "reference": "Au"},
    {"input": "What gas do plants absorb from the atmosphere?", "reference": "Carbon dioxide"},
    {"input": "How many bones are in the adult human body?", "reference": "206"},
    {"input": "What is the speed of light in vacuum (km/s, rounded)?", "reference": "300000"},

    # Math / logic (short exact)
    {"input": "2+2?", "reference": "4"},
    {"input": "12 * 12?", "reference": "144"},
    {"input": "Square root of 81?", "reference": "9"},
    {"input": "What is 15% of 200?", "reference": "30"},
    {"input": "If x=3, what is 2x+5?", "reference": "11"},

    # History / literature
    {"input": "Who wrote Hamlet?", "reference": "Shakespeare"},
    {"input": "In which year did World War II end?", "reference": "1945"},
    {"input": "Who was the first president of the United States?", "reference": "George Washington"},

    # Yes/No (format-sensitive)
    {"input": "Is the Earth flat? Answer yes or no.", "reference": "No"},
    {"input": "Is 17 a prime number? Answer yes or no.", "reference": "Yes"},
]


SUMMARIZATION_DATASET = [
    {
        "input": (
            "A new study reported that regular exercise improves cardiovascular health. "
            "Researchers tracked 5,000 adults over ten years and found lower rates of heart disease. "
            "They recommend at least 150 minutes of moderate activity per week. "
            "The study also noted improved sleep quality and reduced stress among participants."
        ),
        "reference": (
            "A 10-year study of 5,000 adults found regular exercise reduces heart disease risk and improves sleep and stress, "
            "recommending 150 minutes of moderate activity weekly."
        ),
    },
    {
        "input": (
            "The city council approved a budget to expand public transportation. "
            "The plan adds three new bus routes, increases service frequency during peak hours, and upgrades stations for accessibility. "
            "Officials expect reduced traffic congestion over the next two years. "
            "Construction starts in March and is funded by a mix of local taxes and federal grants."
        ),
        "reference": (
            "The city council funded a transit expansion adding bus routes, increasing peak service, and upgrading stations, "
            "aiming to cut congestion within two years using local taxes and federal grants."
        ),
    },
    {
        "input": (
            "A software company released a security update after researchers discovered a vulnerability affecting millions of devices. "
            "The flaw allowed attackers to execute code remotely under certain conditions. "
            "Users are urged to update immediately, and administrators should review logs for suspicious activity. "
            "The company credited an external research team and announced a bug bounty increase."
        ),
        "reference": (
            "A security update was released to fix a remote code execution vulnerability affecting millions of devices, "
            "and users are urged to update and check logs; the company credited researchers and increased its bug bounty."
        ),
    },
    {
        "input": (
            "A major retailer reported quarterly earnings that exceeded analyst expectations. "
            "Revenue grew 8% year-over-year, driven by online sales and improved inventory management. "
            "However, executives warned that shipping costs remain high and could impact margins. "
            "The company plans to expand same-day delivery to 20 more cities."
        ),
        "reference": (
            "The retailer beat earnings expectations with 8% revenue growth from online sales and inventory improvements, "
            "while warning high shipping costs may pressure margins and planning to expand same-day delivery."
        ),
    },
    {
        "input": (
            "Scientists tested a new battery material that charges faster and retains capacity longer. "
            "In lab conditions, prototypes reached 80% charge in under 10 minutes and maintained 90% capacity after 1,000 cycles. "
            "Researchers cautioned that scaling manufacturing remains challenging. "
            "They will next test performance under extreme temperatures."
        ),
        "reference": (
            "Researchers demonstrated a battery material that charges to 80% in under 10 minutes and keeps 90% capacity after 1,000 cycles, "
            "though manufacturing scale-up remains difficult and further testing is planned."
        ),
    },
    {
        "input": (
            "A school district introduced a new reading program to address declining literacy scores. "
            "The initiative includes teacher training, updated classroom materials, and after-school tutoring. "
            "Early pilot results showed improved reading comprehension among third graders. "
            "The district will evaluate the program again at the end of the academic year."
        ),
        "reference": (
            "The district launched a reading program with teacher training, new materials, and tutoring, "
            "and early pilots improved third-grade comprehension, with further evaluation planned later in the year."
        ),
    },
    {
        "input": (
            "A national park announced temporary trail closures due to wildfire risk. "
            "Officials cited dry conditions, high winds, and limited firefighting resources. "
            "Visitors are asked to follow updated maps, avoid prohibited areas, and report smoke sightings. "
            "The park will reassess conditions weekly."
        ),
        "reference": (
            "The national park temporarily closed trails because of elevated wildfire risk from dry, windy conditions, "
            "asking visitors to follow updates and report smoke while officials reassess weekly."
        ),
    },
    {
        "input": (
            "A hospital implemented an AI-assisted triage system to prioritize emergency room cases. "
            "The system analyzes symptoms and vital signs to estimate risk, but clinicians make final decisions. "
            "Early results suggest shorter wait times for critical patients. "
            "Administrators emphasized that patient privacy safeguards were built into deployment."
        ),
        "reference": (
            "A hospital deployed an AI-assisted triage tool that estimates risk from symptoms and vital signs to help prioritize ER cases, "
            "with clinicians deciding and early results showing shorter waits for critical patients alongside privacy safeguards."
        ),
    },
    {
        "input": (
            "An airline introduced new rules for carry-on baggage to reduce boarding delays. "
            "Passengers may bring one standard carry-on and one personal item, with stricter size enforcement at the gate. "
            "The airline said the policy aims to improve on-time performance. "
            "Some customers expressed concerns about added fees for oversized bags."
        ),
        "reference": (
            "The airline tightened carry-on rules to reduce boarding delays by enforcing size limits and allowing one carry-on plus one personal item, "
            "aiming to improve on-time performance amid customer concerns about fees."
        ),
    },
    {
        "input": (
            "A coastal city began installing flood barriers after repeated storm surges damaged homes and roads. "
            "Engineers designed movable gates for key entry points and upgraded drainage systems. "
            "The project is expected to take 18 months and cost $45 million. "
            "Officials said the measures should reduce damage during severe weather."
        ),
        "reference": (
            "The city is installing movable flood barriers and upgrading drainage after storm surge damage, "
            "with an 18-month, $45 million project aimed at reducing severe-weather impacts."
        ),
    },
]
