from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("Modelo GPT-2 carregado com sucesso!")

input_text = "Artificial intelligence is transforming various industries, including healthcare, education, and technology. One example is"

inputs = tokenizer(input_text, return_tensors="pt", padding=True)

output = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nTexto gerado:\n", generated_text)
