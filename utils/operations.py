import os


def run_command(command):
    result = os.system(command=command)
    return result


def write_file(input_data):
    filename = input_data["filename"]
    code = input_data["code"]
    try:
        with open(filename, "w") as f:
            f.write(code)
        return f"{filename} written successfully."
    except Exception as e:
        return str(e)
