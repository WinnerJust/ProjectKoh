import pandas as pd
import random
import re
import torch
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import sklearn.tree

import pydotplus

import networkx
from networkx.readwrite import json_graph
import pydot

import networkx as nx
import matplotlib.pyplot as plt

from graphviz import Digraph, Source

from IPython.display import display






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






with open('../data/logits_names_map.txt', 'r') as file:
    logits_names_map = json.load(file)






def gpt(text):
    inputs = tokenizer_loaded(text=text, return_tensors="pt", padding=True, truncation=True,  max_length=max_length)
    #print(inputs)
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
    #print("Predicted Labels:", predicted_labels)
    
    return predicted_labels









df = pd.read_csv("../data/small_train_patients.csv")
df = df.rename(columns={'PATHOLOGY': 'prognosis'})
df = df.iloc[:, 3:]

for col in df.columns[1:]:
  df[col] = df[col].replace('False', 0)

df.iloc[:, 1:] = df.iloc[:, 1:].astype('int64')



first_column = df.pop('prognosis')
numpy_array = df.values

#Putting back the column (after we converted df without classs labels to numpy x array)
#We need this column for our start() function
df['prognosis'] = first_column

labels = first_column.values
feature_names = df.columns.to_numpy()




X_train, X_test, y_train, y_test = train_test_split(numpy_array, labels, test_size=0.2, random_state=42)






def _entropy(left_y, right_y):
    p_left, p_right = len(left_y), len(right_y)
    total = p_left + p_right

    unique_left, counts_left = np.unique(left_y, return_counts=True)
    unique_right, counts_right = np.unique(right_y, return_counts=True)

    prob_left = counts_left / p_left
    prob_right = counts_right / p_right


    #entropy_left = -np.sum(prob_left * np.log2(prob_left))
    #entropy_right = -np.sum(prob_right * np.log2(prob_right))
    entropy_left = -np.sum(prob_left * np.log2(prob_left + 1e-10))
    entropy_right = -np.sum(prob_right * np.log2(prob_right + 1e-10))

    entropy = (p_left / total) * entropy_left + (p_right / total) * entropy_right
    return entropy
    
    
    
    
    

dot = Digraph()
node_counter = 0

def generate_unique_id():
    global node_counter
    node_counter += 1
    return str(node_counter)

def draw_tree(tree, filename):
    global dot
    global node_counter

    current_id = generate_unique_id()
    dot.node(name=current_id, label=str(tree['feature']) + ' ' + str(tree['index']) + ' ' + str(tree['value']) + '\nSamples:' + str(tree['samples']) + '\nSamples check:' + str(tree['samples_check'])) # + '\n' + str(tree['uniques']))

    if isinstance(tree['left'], dict):
        #left_id = current_id + 1 #generate_unique_id()
        left_id = str(node_counter + 1)
        dot.edge(current_id, left_id)
        draw_tree(tree['left'], filename)
    else:
        left_id = generate_unique_id()
        dot.node(name=left_id, label=str(tree['left']))
        dot.edge(current_id, left_id)

    if isinstance(tree['right'], dict):
        right_id = str(node_counter + 1) #generate_unique_id()
        #Exactly this ID will be generated in draw_tree call
        dot.edge(current_id, right_id)
        draw_tree(tree['right'], filename)
    else:
        #Here we generate right here, because there is no draw_tree call which will generate ID
        right_id = generate_unique_id()
        dot.node(name=right_id, label=str(tree['right']))
        dot.edge(current_id, right_id)
        
        
        
        
def convert_to_title_case(name):
    words = name.split('_')
    words[0] = words[0].capitalize()
    return ' '.join(words)
        
        
        
        
START_WORDS = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'can', 'does', 'do', 'which', 'whose', 'whom', 'would', 'will', 'should', 'could', 'are', 'were', 'has', 'have', 'had', 'may', 'might', 'was']

def isQuestion(sentence):
    return sentence.endswith('?') or any(sentence.lower().startswith(word) for word in START_WORDS)


    
    
    
debug = True










def start():
  random_row_index = 333
  #random_row_index = random.randint(0, len(df) - 1)
  #print(random_row_index)
  random_row = df.iloc[random_row_index]
  print(random_row)

  prognosis = random_row["prognosis"]
  print("SYSTEM: Our patient has the following disease:", prognosis)

  symptoms = random_row[random_row == 1].index.tolist()
  print("SYSTEM: And he has the following symptoms")
  print(symptoms)

  random_symptom_orig = symptoms[5]
  #random_symptom_orig = random.choice(symptoms)
  random_symptom = convert_to_title_case(random_symptom_orig)
  #print(random_symptom)

  query = f"Please write one sentence to say hello to doctor on online consultation "\
  f"and tell him that I have {random_symptom}. But don't mention online consultation. And do not ask advice."
  #print(query)

  #first_message = gpt(query)
  #print(first_message)
  print("Hello, Doctor! I have", random_symptom)





  column_names = df.columns.tolist()
  column_names = column_names[:-1]




  index = column_names.index(random_symptom_orig)

  mask = X_train[:, index] == 1

  #This is for finding best subtree before even conversating
  best_sub_tree_x = X_train[mask]
  best_sub_tree_y = y_train[mask]

  feature_history = np.array([])

  best_graph = Digraph()

  current_node_id = 0
  best_graph.node(name=str(current_node_id), label=random_symptom)
  best_graph.edge(str(current_node_id), str(current_node_id + 1), label="Yes")




  #Finding best tree possible
  b_index, b_score = 999, 999
  our_score = None


  for depth in range(15):
      #If in current best_sub_tree entropy is zero, that is all items are of one class
      #We finish
      if np.all(best_sub_tree_y == best_sub_tree_y[0]):
          break

      for index in range(best_sub_tree_x.shape[1]):
          #print("Trying index", index)

          #print(column_names[index])

          feature_name_part = column_names[index].split("_@_")[0]

          if feature_name_part in ['body_pain_location', 'pain_radiation', 'pain_characteristics', 'pain_sudden_appearance', 'pain_intensity', 'pain_localization']:
              if not np.isin('pain', feature_history):
                  continue
          elif feature_name_part in ['skin_lesion_location', 'lesion_color', 'lesion_pain_intensity', 'lesion_swelling', 'lesion_itching', 'lesion_larger_than_1cm', 'lesion_peeling']:
              if not np.isin('skin_lesions', feature_history):
                  continue

          value = 0
          left_mask = best_sub_tree_x[:, index] == value
          left_y = best_sub_tree_y[left_mask]
          right_y = best_sub_tree_y[~left_mask]

          #print(len(left_y))
          #print(len(right_y))


          ent = _entropy(left_y, right_y)
          #print("Entropy for", column_names[index], "is", ent)

          if ent < b_score:
              #left_x = x[left_mask]
              #right_x = x[~left_mask]
              b_index, b_score = index, ent


      #In the loop we calculate only splits of y (for efficiency)
      #Here we split everything because we need everything
      left_mask = best_sub_tree_x[:, b_index] == 0
      left_x = best_sub_tree_x[left_mask]
      right_x = best_sub_tree_x[~left_mask]
      left_y = best_sub_tree_y[left_mask]
      right_y = best_sub_tree_y[~left_mask]

      best_sub_tree_x = right_x if column_names[b_index] in symptoms else left_x
      best_sub_tree_y = right_y if column_names[b_index] in symptoms else left_y

      best_graph.node(name=str(current_node_id + 1), label=column_names[b_index])
      #We create edge coming from new node, not into new node
      best_graph.edge(str(current_node_id + 1), str(current_node_id + 1 + 1), label="Yes" if column_names[b_index] in symptoms else "No")

      current_node_id += 1


  #So we finish either by reaching 15 questions, or having all samples of one class
  #Either way we count most popular class
  unique_outcomes, counts = np.unique(best_sub_tree_y, return_counts=True)
  best_prognosis = unique_outcomes[np.argmax(counts)]

  best_graph.node(name=str(current_node_id + 1), label=best_prognosis)
  #best_graph.edge(current_node_id, current_node_id + 1, label="Yes" if column_names[b_index] in symptoms else "No")



  best_graph.render('best_question_chain', view=True)






  sub_tree_x = X_train[mask]
  sub_tree_y = y_train[mask]

  next_sub_tree_x = None
  next_sub_tree_y = None

  feature_history = np.array([])


  user_graph = Digraph()

  current_node_id = 0
  user_graph.node(name=str(current_node_id), label=random_symptom)
  user_graph.edge(str(current_node_id), str(current_node_id + 1), label="Yes")

  #Brought those to here to reach not only when we get symptom but also when we get prognosis
  b_index, b_score = 999, 999
  our_score = None






  got_prognosis = False

  while not got_prognosis:
    inp = input()

    number = gpt(inp)

    if number:
        id = int(number)

        if isQuestion(inp):
          #We got symptom
          symptom_queried = logits_names_map[id]
          print("SYSTEM: Symptom queried:", symptom_queried)



          #Here should be tree step and question evaluation

          #b_index, b_value, b_score, b_groups = 999, 999, 999, None
          #b_index, b_score = 999, 999
          #our_score = None


          #sub_tree_x = X_train
          #sub_tree_y = y_train

          #next_sub_tree_x = None
          #next_sub_tree_y = None


          #print(sub_tree_x.shape)
          #print(sub_tree_y.shape)

          #print(sub_tree_x[0]) #Ok this is shuffled
          #print(df.head(1))

          for index in range(sub_tree_x.shape[1]):
              #print("Trying index", index)

              #print(column_names[index])

              feature_name_part = column_names[index].split("_@_")[0]

              if feature_name_part in ['body_pain_location', 'pain_radiation', 'pain_characteristics', 'pain_sudden_appearance', 'pain_intensity', 'pain_localization']:
                  if not np.isin('pain', feature_history):
                      continue
              elif feature_name_part in ['skin_lesion_location', 'lesion_color', 'lesion_pain_intensity', 'lesion_swelling', 'lesion_itching', 'lesion_larger_than_1cm', 'lesion_peeling']:
                  if not np.isin('skin_lesions', feature_history):
                      continue

              value = 0
              left_mask = sub_tree_x[:, index] == value
              left_y = sub_tree_y[left_mask]
              right_y = sub_tree_y[~left_mask]

              #print(len(left_y))
              #print(len(right_y))


              ent = _entropy(left_y, right_y)
              #print("Entropy for", column_names[index], "is", ent)

              if ent < b_score:
                  #left_x = x[left_mask]
                  #right_x = x[~left_mask]
                  b_index, b_score = index, ent

              if column_names[index] == symptom_queried and ent:
                  our_score = ent

                  left_x = sub_tree_x[left_mask]
                  right_x = sub_tree_x[~left_mask]
                  next_sub_tree_x = right_x if symptom_queried in symptoms else left_x
                  next_sub_tree_y = right_y if symptom_queried in symptoms else left_y

          sub_tree_x = next_sub_tree_x
          sub_tree_y = next_sub_tree_y

          print("SYSTEM: Best symptoms that could be queried that this point:", column_names[b_index])
          print("SYSTEM: Its entropy is:", b_score)

          print("SYSTEM: You queried:", symptom_queried)
          print("SYSTEM: Its entropy is:", our_score)

          #This object goes to split function and gets saved as a node using links from other nodes
          #And arguments of this function we get back from split function, that is feature_history
          #return {'index': b_index, 'value': b_value, 'left_x': b_left_x, 'left_y': b_left_y, 'right_x': b_right_x, 'right_y': b_right_y, 'feature': feature_names[b_index], 'entropy': ent,
          #        'feature_history': np.append(feature_history, feature_names[b_index])}




          #Updating user graph

          #Drawing alternative (best) question
          #Here we have to draw edge from previous node
          user_graph.node(name=str(1000 - current_node_id + 1), label="Best question: " + column_names[b_index])
          user_graph.edge(str(current_node_id), str(1000 - current_node_id + 1), label="Yes" if column_names[b_index] in symptoms else "No")

          #Drawing user's question
          user_graph.node(name=str(current_node_id + 1), label=symptom_queried)
          user_graph.edge(str(current_node_id + 1), str(current_node_id + 1 + 1), label="Yes" if symptom_queried in symptoms else "No")

          current_node_id += 1




          if symptom_queried in symptoms:
            message = 'Patient indeed has this symptom/problem.'
          else:
            message = "Patient doesn't have this symptom/problem."

          print("SYSTEM:", message)
          
          print("Yes" if symptom_queried in symptoms else "No")



        else:
          #We got prognosis
          got_prognosis = True

          prognosis_predicted = None
          
          unique_prognoses = df["prognosis"].unique().tolist()
          
          for prognose in unique_prognoses:
            if prognose.lower() in inp:
                prognosis_predicted = prognose
                
          print("SYSTEM: Predicted prognosis:")
          print(prognosis_predicted)
          
          if not prognosis_predicted:
            got_prognosis = False
            print("What exactly do you mean?")
            continue

          if prognosis_predicted == prognosis:
            print("SYSTEM: correct prediction")
          else:
            print("SYSTEM: you failed. Correct prognosis is", prognosis)



          #What about best prognosis here?
          #Like what could you guess based on remaining features
          #Sound like why not

          unique_outcomes, counts = np.unique(sub_tree_y, return_counts=True)
          best_prognosis = unique_outcomes[np.argmax(counts)]

          user_graph.node(name=str(1000 - current_node_id + 1), label="Best diagnosis could be made now: " + best_prognosis)
          user_graph.edge(str(current_node_id), str(1000 - current_node_id + 1), label="Yes" if column_names[b_index] in symptoms else "No")

          user_graph.node(name=str(current_node_id + 1), label=prognosis_predicted)
          #user_graph.edge(current_node_id, current_node_id + 1, label="Yes" if column_names[b_index] in symptoms else "No")



          user_graph.render('user_question_chain', view=True)

  #print(user_graph)

start()







