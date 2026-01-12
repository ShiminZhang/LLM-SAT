
from llmsat.pipelines.chatgpt_data_generation import generate_team_data


generate_team_data(
    designer_prompt_path="./data/prompts/ae_prompt.txt",
    variant_prompt_path="./data/prompts/variant_prompt.txt",
    code_prompt_template_path="./data/prompts/kissat_mab_code.txt",
    generation_tag="test_teamleader",
    n_leaders=2,
    m_variants_per_leader=1,
    model="gpt-4o"
)
