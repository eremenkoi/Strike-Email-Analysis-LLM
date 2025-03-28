#!/usr/bin/env python3
import subprocess

def run_gemma3(prompt_text):
    """
    Runs the Gemma3:12b model via Ollama using a subprocess call.
    Uses a list for arguments to avoid shell quoting issues.
    """
    command = ["ollama", "run", "gemma3:12b", prompt_text]
    print("Running command:", " ".join(command))

    try:
        # Pass stdin=subprocess.DEVNULL to avoid waiting for STDIN input
        result = subprocess.run(command, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e.stderr)
        return None

def main():
    # Define a simple test prompt.
    test_prompt = "Hello, Gemma3! Please respond with a friendly greeting."
    print("Testing Gemma3 with prompt:")
    print(test_prompt)

    # Run the prompt through Gemma3.
    response = run_gemma3(test_prompt)
    if response is None:
        print("Test failed: No response from Gemma3.")
    else:
        print("Response from Gemma3:")
        print(response)

if __name__ == "__main__":
    main()
