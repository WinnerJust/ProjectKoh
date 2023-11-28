import torch
print(torch.__version__)


with open('../data/logits_names_map.txt', 'r') as file:
    logits_names_map = json.load(file)
    


from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
                          
                          
                          
                          
                          
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 60
                          
                          
                          
# Define the directory where you saved the model
saved_model_dir = '../model/'

# Load the model, configuration, and tokenizer
#model_loaded = GPT2Model.from_pretrained(saved_model_dir)
tokenizer_loaded = GPT2Tokenizer.from_pretrained(saved_model_dir)





#print('Loading configuraiton...')
#model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=422)

# Get model's tokenizer.
#print('Loading tokenizer...')
#tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer_loaded.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer_loaded.pad_token = tokenizer_loaded.eos_token


# Get the actual model.
#print('Loading model...')
model_loaded = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=saved_model_dir, config=model_config)

# resize model embedding to match new tokenizer
model_loaded.resize_token_embeddings(len(tokenizer_loaded))

# fix model padding token id
model_loaded.config.pad_token_id = model_loaded.config.eos_token_id

# Load model to defined device.
model_loaded.to(device)
print('Model loaded to `%s`'%device)






# Your loaded model, tokenizer, and device are assumed to be available.
model_loaded.to(device)


# Sample text for prediction
sample_text = "Your input text goes here."
sample_text = "Do you have weight loss?"
sample_text = "Have you gained some weight recently?"

# Tokenize the input text
#inputs = tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

print("max_length:", max_length)

inputs = tokenizer_loaded(text=sample_text, return_tensors="pt", padding=True, truncation=True,  max_length=max_length)
print(inputs)
        # Update the inputs with the associated encoded labels as tensor.
        #inputs.update({'labels':torch.tensor(labels)})

# Move the inputs to the defined device
#inputs = inputs.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

#batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

# Perform prediction
with torch.no_grad():
    model_loaded.eval()
    outputs = model_loaded(**inputs)

# Get the predicted labels/logits
logits = outputs.logits
predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

# Print predicted labels
print("Predicted Labels:", predicted_labels)

'''

target_int = predicted_labels[0]

result = None
for key, value in labels_ids.items():
    if value == target_int:
        result = key
        break

if result is not None:
    print(f"The string corresponding to {target_int} is: {result}")
else:
    print(f"No string found for the integer {target_int}")
    
    '''