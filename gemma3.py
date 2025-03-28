import subprocess


def run_gemma3(prompt_text):
    """
    Runs the Ollama Gemma3:12b model with the given prompt.
    Returns the model's output as a string.
    """
    # Wrap the prompt text in quotes so that the entire string is passed as a single argument.
    command = 'ollama run gemma3:12b "{}"'.format(prompt_text)
    print("Running command:", command)  # Debug print

    # Run the command with shell=True on Windows
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    # Debug prints for troubleshooting
    if result.returncode != 0:
        print("Error running Ollama:", result.stderr)
        return None

    print("Return code:", result.returncode)
    print("Stdout:", result.stdout)
    print("Stderr:", result.stderr)

    return result.stdout


if __name__ == "__main__":
    # Provide a concrete example conversation.
    conversation_text = "Customer: I need a quote for 100 units. Sales: Sure, our price is $500 each."

    # Wrap the conversation text in quotes in the prompt.
    prompt = 'Extract sales quote details from the following conversation: "{}"'.format(conversation_text)

    # Call the function and print the output.
    output = run_gemma3(prompt)
    if output:
        print("Output from Gemma3:12b:")
        print(output)
