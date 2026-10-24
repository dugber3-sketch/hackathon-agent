import requests
import subprocess
import tempfile
import os

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# 1. Define a local tool to compile and test C code
def compile_and_run_c(c_code: str) -> str:
    print("\n[Compiling and testing C code locally...]")
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "main.c")
        exe_file = os.path.join(tmpdir, "main")
        
        with open(c_file, "w") as f:
            f.write(c_code)
            
        compile_proc = subprocess.run(["gcc", c_file, "-o", exe_file], capture_output=True, text=True)
        if compile_proc.returncode != 0:
            return f"Compilation Error:\n{compile_proc.stderr}"
            
        run_proc = subprocess.run([exe_file], capture_output=True, text=True, timeout=5)
        return f"Program Output:\n{run_proc.stdout}\n{run_proc.stderr}".strip()

# 2. Define the tool schema for Ollama
tools = [{
    "type": "function",
    "function": {
        "name": "compile_and_run_c",
        "description": "Compiles and runs a C program using gcc and returns the output or errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "c_code": {"type": "string", "description": "The complete C source code to compile and run."}
            },
            "required": ["c_code"]
        }
    }
}]

messages = [{
    "role": "user",
    "content": "Write a very short, funny C program demonstrating how include guards work. Use the tool to compile and run it."
}]

try:
    print("Sending request to local gemma4:e2b agent...")
    response = requests.post(OLLAMA_CHAT_URL, json={
        "model": "gemma4:e2b",
        "messages": messages,
        "tools": tools,
        "stream": False
    })
    response.raise_for_status()
    message = response.json()["message"]
    messages.append(message)

    # 3. Check if gemma4:e2b decided to call the tool
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            code = tool_call["function"]["arguments"]["c_code"]
            print(f"\n--- Generated C Code ---\n{code}")
            
            tool_output = compile_and_run_c(code)
            print(f"\n--- Execution Result ---\n{tool_output}")
            
            # Send the execution result back to gemma4:e2b
            messages.append({"role": "tool", "content": tool_output})
            
        final_res = requests.post(OLLAMA_CHAT_URL, json={
            "model": "gemma4:e2b",
            "messages": messages,
            "stream": False
        }).json()
        print("\n--- Final Agent Response ---")
        print(final_res["message"]["content"])
    else:
        print("\n--- Gemma Output ---")
        print(message.get("content"))

except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to Ollama. Make sure 'ollama serve' is running.")