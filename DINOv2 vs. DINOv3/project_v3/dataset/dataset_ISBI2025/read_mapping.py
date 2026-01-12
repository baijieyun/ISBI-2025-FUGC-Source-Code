import json

def get_mapping_json(path="./final_mapping.json",mode="train",isLabeled=True):
    file = open(path,"r")
    json_file = json.load(file)
    if mode == "train":
        return json_file[mode]["labeled"] if isLabeled else json_file["train"]["unlabeled"]
    else:
        return json_file[mode]

def reverse_mapping(mapping:dict):
    reversed_mapping = { v:k for k,v in mapping.items()}
    return reversed_mapping


if __name__ == "__main__":
    mapping_path = "./final_mapping.json"
    mode = "train"
    isLabeled = True
    mapping = get_mapping_json(mapping_path,mode,isLabeled)
    reversed_mapping = reverse_mapping(mapping)
    print(reversed_mapping)