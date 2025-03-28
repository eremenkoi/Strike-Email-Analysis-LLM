from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the manually downloaded Llama 2 model
LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\good_model"

try:
    print(f"🔄 Loading Llama 2 from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, device_map="auto")
    print("✅ Llama 2 loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
