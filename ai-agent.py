from openai import OpenAI
from dotenv import load_dotenv
from utils.operations import run_command, write_file
import json
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

available_tools = {
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input and executes it on the system",
    },
    "write_file": {
        "fn": write_file,
        "description": "Writes code to the specified file",
    },
}

with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

messages = [{"role": "system", "content": system_prompt}]

while True:
    user_query = input("> ")
    if user_query.strip().lower() == "clear":
        break
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type": "json_object"},
            n=1,
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        # print(parsed_output)

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get("content")}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "output": output}),
                    }
                )
                continue

        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get("content")}")
            break
