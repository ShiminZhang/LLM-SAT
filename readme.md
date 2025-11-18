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
