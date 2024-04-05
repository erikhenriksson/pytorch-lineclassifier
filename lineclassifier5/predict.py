import json


def stream_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def run(cfg):

    # Example usage
    file_path = "path_to_your_large_file.json"
    for json_object in stream_json_file(file_path):
        # process each json_object here
        print(json_object)
