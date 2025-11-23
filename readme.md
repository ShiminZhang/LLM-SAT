fir

Input: Algorithms output from ChatGPT in jsonl

Output:
1. Codes for the algorithms
2. Solvers
3. Par2 scores

Step:
1. Convert algorithm output to algorithm string: python src.llmsat.data.algorithm_parse.py --input input.jsonl
2. Store algorithm string: 
    Storing done by src.llmsat.utils.aws.py: 
        2 tables, one for code, one for algorithm
            AlgorithmResult, CodeResult
3. Generate code based on algorithm
    src.llmsat.evaluation.coder.py
        all codes will be stored into aws by coder
4. Evaluate the code 
    done at src.llmsat.pipelines.evaluation.py
    1. build
    2. run


# To setup
pip install -r requirements.txt
PYTHONPATH=./src:$PYTHONPATH

## setup aws
run everytime or insert into .bashrc or use .env: export DB_PASS="Damn123," 

## 

1. Try Kissat-MAB and AE-MAB, debug on-the-fly 
2. After we have the data: algorithm-code-par2
    start finetuning DPO+RLSF
    Options:
        1. Single family benchmark
            - can even do online training
        2. Family-aware

# ChatGPT pipeine:

## generation
to use, run: python src/llmsat/pipelines/chatgpt_data_generation.py

a new table CHATGPT_DATA_GENERATION_TABLE has added in database to keep track of the chatgpt results

generate_data() function for generating the algorithms and corresponding codes. All result saved to database.
- **generation_tag** argument takes string input, and will mark all new algorithms generated with current config with that tag in CHATGPT_DATA_GENERATION_TABLE
- n_algorithms parameter sets the number of algorithms to generate. We could start by 50 for now before decide which solver to use.
- designer_prompt_path points to the prompt for algorithm generation, you may want to have a different prompt for Kissat-MAB and AE-MAB
- code_prompt_template_path points to the prompt for code generation, this has to be a different prompt for Kissat-MAB and AE-MAB
- model: try use gpt-4o and gpt-5, let's see which gives better result. gpt-4o is used by default.

## retrieve
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    example in print_generation_result() function.

## evaluation
sbatch scripts/start_evaluation.sh

- real logic is in python src/llmsat/pipelines/evaluation.py --run_all

you need to set correct generation_tag in the main() function (or add it to argparse and parse it in). For now it's:


    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, "chatgpt_data_generation_gpt5_2")

The function will find corresponding algorithms and evaluate them.
