from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\models-tiiuae--falcon-7b-instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, device_map="auto")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", str(e))
