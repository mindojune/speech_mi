from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# intialize LLM + Speech Adapter

# use GeneZC/MiniChat-2-3B

model_name = "GeneZC/MiniChat-2-3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ask for user input and get continuation
input_text = input("Enter your prompt: ")
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
#