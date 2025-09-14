import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

class DataManager:
    def __init__(self, data_path="/pscratch/sd/s/shimin/verltest/dataset/dataset_preprocessed.jsonl"):
        # self.data = create_test_data()
        self.data_path = data_path
        self.train_dataset = self.load_data()
        pass
    def load_data(self):
        result = []
        with open(self.data_path, "r") as f:
            for line in f:
                result.append(json.loads(line))
        return result

    def get_data(self, size):
        return self.train_dataset.take(size)

def preprocess_data(original_data_point: dict):
    N_of_heuristic = len(original_data_point["prompts"].keys())
    N_for_winner = len(original_data_point["winner_solver"].keys())
    N_for_loser = len(original_data_point["loser_solver"].keys())
    N_of_heuristic = min(N_of_heuristic, N_for_winner, N_for_loser)
    data = []
    for index in range(N_of_heuristic):

        decomposed_data_point = {
            "prompt" : None,
            "chosen" : None,
            "rejected" : None
        }
        system_prompt = original_data_point["prompts"][str(index + 1)][0]["content"]
        user_prompt = original_data_point["prompts"][str(index + 1)][1]["content"]
        decomposed_data_point["prompt"] = f"{system_prompt}\n{user_prompt}"
        decomposed_data_point["chosen"] = original_data_point["winner_solver"][str(index + 1)]
        decomposed_data_point["rejected"] = original_data_point["loser_solver"][str(index + 1)]
        data.append(decomposed_data_point)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/dataset_raw.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    # data = DataManager(args.data_path).get_data(args.limit)
    if args.preprocess:
        with open(args.data_path, "r") as f:
            data = json.load(f)
            dpo_datas = []
            count = 0
            for data_point in tqdm(data):
                dpo_data =preprocess_data(data_point)
                dpo_datas.append(dpo_data)
                count += 1
                if count >= args.limit:
                    break
            with open("dataset/dataset_preprocessed.jsonl", "w") as f:
                for data_points in dpo_datas:
                    for data_point in data_points:
                        json.dump(data_point, f)
                        f.write("\n")

    if args.test:
        data = []
        with open("dataset/dataset_preprocessed.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
                break
            print(len(data[0]))
            print(data[0]["prompt"])
            print(data[0]["chosen"])
            print(data[0]["rejected"])

if __name__ == "__main__":
    main()