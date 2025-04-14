from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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

system_prompt = f"""
    You are an helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - If the user wants you to write the code then when you write the code in the particular file then don't add it as string.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string" or "code",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - run_command: Takes a command as input to execute on system and returns output
    - write_file: Writes code into the specified file. Takes {{ "filename": "file_name", "code": "code to write" }}

    Example 1:
    User Query: can you add a file in the dir D:/VScode named hello.txt?
    Output: {{ "step": "plan", "content": "The user is interested in adding a file at a given directory D:/VScode/" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call run_command" }}
    Output: {{ "step": "action", "function": "run_command", "input": "hello.txt" }}
    Output: {{ "step": "observe", "output": "hello.txt added" }}
    Output: {{ "step": "output", "content": "hello.txt file added in the D:/VScode/ directory." }}

    Example 2:
    User Query: Write a Python function called add that takes two numbers and returns the sum in file utils.py.
    Output: {{ "step": "plan", "content": "The user wants to write a Python function in the file utils.py" }}
    Output: {{ "step": "plan", "content": "I need to write the actual code in the utils.py file" }}
    Output: {{
        "step": "action",
        "function": "write_file",
        "input": {{
            "filename": "utils.py",
            "code":
            def add(a, b):
                return a + b
        }}
    }}

    Example 3:
    User Query: Please create a function in app.js that logs 'Hello World' to the console.
    Output: {{ "step": "plan", "content": "The user wants to write a JavaScript function in the file app.js" }}
    Output: {{ "step": "plan", "content": "I will write the code directly in app.js" }}
    Output: {{
        "step": "action",
        "function": "write_file",
        "input": {{
            "filename": "app.js",
            "code": 
            function greet() {{
                console.log('Hello World');
            }}
        }}
    }}
"""

messages = []

while True:
    user_query = input("> ")
    if user_query.strip().lower() == "clear":
        break
    messages.append(json.dumps({"step": "query", "content": user_query}))

    while True:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=messages,
        )
        cleaned_text = re.sub(
            r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE
        ).strip()

        parsed_outputs = []
        try:
            parsed_outputs.append(json.loads(cleaned_text))
        except json.JSONDecodeError:
            for line in cleaned_text.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed_outputs.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("❌ Could not parse line:", line)

        for parsed_output in parsed_outputs:
            messages.append(json.dumps(parsed_output))

            if parsed_output.get("step") == "plan":
                print(f"🧠: {parsed_output.get('content')}")
                continue

            if parsed_output.get("step") == "action":
                tool_name = parsed_output.get("function")
                tool_input = parsed_output.get("input")

                if available_tools.get(tool_name):
                    output = available_tools[tool_name]["fn"](tool_input)
                    messages.append(json.dumps({"step": "observe", "output": output}))
                    break

            if parsed_output.get("step") == "output":
                print(f"🤖: {parsed_output.get('content')}")
                break
