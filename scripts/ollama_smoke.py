from src.prompt.representation import Prompt
from src.evaluation.ollama_llm import OllamaLLM
from src.evaluation.metrics import ExactMatchMetric
from src.evaluation.evaluator import Evaluator
from src.evaluation.llm_interface import LLMConfig

dataset = [{"input": "Capital of France?", "reference": "Paris"}]

p = Prompt(
    task="qa",
    instruction="Answer the question.",
    constraints=["Answer with only the final answer (no extra text)."],
    output_format="plain",
    style="concise",
)

llm = OllamaLLM()  # uses OLLAMA_HOST and OLLAMA_MODEL env vars if set
cfg = LLMConfig(temperature=0.0, timeout_s=60.0)
evaluator = Evaluator(llm=llm, metric=ExactMatchMetric(), llm_config=cfg)

res = evaluator.score_prompt(p, dataset)
print("score:", res.quality_score)
print("prediction:", res.per_example[0].prediction)
