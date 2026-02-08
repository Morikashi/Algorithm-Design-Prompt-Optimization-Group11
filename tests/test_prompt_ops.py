from src.prompt.representation import Prompt
from src.prompt.operators import generate_neighbors, add_constraint


def test_prompt_render_deterministic():
    p = Prompt(
        task="qa",
        instruction="Answer the question.",
        constraints=["Answer with only the final answer (no extra text).", "Do not explain your reasoning."],
        output_format="plain",
        style="concise",
        verification=True,
    )
    r1 = p.render()
    r2 = p.render()
    assert r1 == r2
    assert "[TASK: qa]" in r1
    assert "Constraints:" in r1


def test_add_constraint_no_duplicates():
    p = Prompt(task="qa", instruction="Answer the question.")
    p2 = add_constraint(p, "Do not explain your reasoning.")
    p3 = add_constraint(p2, "Do not explain your reasoning.")
    assert len(p2.constraints) == 1
    assert len(p3.constraints) == 1  # should not duplicate


def test_generate_neighbors_non_empty():
    p = Prompt(task="qa", instruction="Answer the question.")
    nbrs = generate_neighbors(p, max_neighbors=10)
    assert len(nbrs) > 0
    assert len(nbrs) <= 10
