import g4f
from g4f.Provider import (
    Bard,
    Bing,
    HuggingChat,
    OpenAssistant,
    OpenaiChat,
    Phind
)


import pandas as pd

df = pd.read_csv("Training (2).csv")

df.head()




def gpt(content):
  response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": content}],
    provider=OpenaiChat,
    access_token=jwt,
    auth=True,
    proxy=your_proxy_here #"http://user:pass@host:port"
  )

  response = response.strip()

  if response.startswith('"') and response.endswith('"'):
    response = response[1:-1]

  return response
  
  
  
def convert_to_title_case(name):
    words = name.split('_')
    words[0] = words[0].capitalize()
    return ' '.join(words)
    
    
    
START_WORDS = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'can', 'does', 'do', 'which', 'whose', 'whom', 'would', 'will', 'should', 'could', 'are', 'were', 'has', 'have', 'had', 'may', 'might', 'was']

def isQuestion(sentence):
    return sentence.endswith('?') or any(sentence.lower().startswith(word) for word in START_WORDS)
    
    
    
    
debug = False











import random
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import sklearn.tree

#import pydotplus

import networkx
from networkx.readwrite import json_graph
import pydot

import networkx as nx
import matplotlib.pyplot as plt



print(g4f.Provider.OpenaiChat.params)

def start():
  #random_row_index = 100
  random_row_index = random.randint(0, len(df) - 1)
  #print(random_row_index)
  random_row = df.iloc[random_row_index]

  prognosis = random_row["prognosis"]
  print("SYSTEM: Our patient has the following disease:", prognosis)

  symptoms = random_row[random_row == 1].index.tolist()
  print("SYSTEM: And he has the following symptoms")
  print(symptoms)

  #random_symptom_orig = symptoms[0]
  random_symptom_orig = random.choice(symptoms)
  random_symptom = convert_to_title_case(random_symptom_orig)
  #print(random_symptom)

  query = f"Please write one sentence to say hello to doctor on online consultation "\
  f"and tell him that I have {random_symptom}. But don't mention online consultation. And do not ask advice."
  #print(query)
  
  #first_message = gpt(query)
  #print(first_message)
  print("Hello, Doctor! I have", random_symptom)



  filtered_df = df[df[random_symptom_orig] == 1]

  data_label = filtered_df.loc[:,"prognosis"]
  data_feature = filtered_df.drop(['prognosis'], axis = 1)


  clf = DecisionTreeClassifier(criterion='entropy')

  clf.fit(data_feature, data_label)



  dot_data = export_graphviz(clf, out_file=None,
                           feature_names=data_feature.columns,
                           #class_names=["0", "1"],
                           #class_names=True,
                           class_names=clf.classes_,
                           filled=True, rounded=True,
                           special_characters=True)

  #Uncomment this to see if it works for you
  #pydot_graph = pydotplus.graph_from_dot_data(dot_data)
  #pydot_graph.write_pdf('tree.pdf')




  graphs = pydot.graph_from_dot_data(dot_data)
  graph = graphs[0]
  graph_netx = networkx.drawing.nx_pydot.from_pydot(graph)
  graph_json = json_graph.node_link_data( graph_netx )
  #print(graph_json)



  our_graph = {
    "nodes": [],
    "links": []
  }

  target_node_id = None
  for node in graph_json['nodes']:
    #print(node)
    if '<entropy = 0.0' in node['label'] and f'class = {prognosis}' in node['label']:
      id = node['id']

      our_node = {
        "label": prognosis,
        "id": int(id)
      }
      our_graph["nodes"].append(our_node)

      target_node_id = id
      break

  print(target_node_id)




  our_id_count = 1000

  #dischromic_patches for example (instant guess of prognosis by the first symptom)
  if not target_node_id:
    our_node = {
      "label": prognosis,
      "id": 0
    }
    our_graph["nodes"].append(our_node)


  else:

    while True:
      source_node_id = -1

      for link in graph_json['links']:
        if target_node_id == link["target"]:
          source_node_id = link["source"]
          break

      #print("source_node_id", source_node_id)

      if source_node_id == -1:
        break #target_node_id is the root



      #So we will be adding crossed out and source node every step
      #At the final step we will add the root as some source node
      for node in graph_json['nodes']:
        if node['id'] == source_node_id:
          our_node = {
            "label": convert_to_title_case(node["label"].split(" ")[0][1:]),
            "id": int(node['id'])
          }
          our_graph["nodes"].append(our_node)

      our_node = {
        "label": "Crossed out",
        "id": our_id_count
      }
      our_graph["nodes"].append(our_node)




      #Determining label for edge (no or yes)
      #But appending no always first
      is_target_left = False

      for link in graph_json['links']:
        if source_node_id == link["source"]:
          if target_node_id == link["target"]:
            is_target_left = True
          break #If found source but target was not target then target is on the right

      #print("is_target_left", is_target_left)

      link_to_crossed = {
        "source": int(source_node_id),
        "target": our_id_count,
        "label": "Yes" if is_target_left else "No"
      }

      link_to_target = {
        "source": int(source_node_id),
        "target": int(target_node_id),
        "label": "No" if is_target_left else "Yes"
      }

      if is_target_left:
        our_graph["links"].append(link_to_target)
        our_graph["links"].append(link_to_crossed)
      else:
        our_graph["links"].append(link_to_crossed)
        our_graph["links"].append(link_to_target)


      our_id_count += 1

      target_node_id = source_node_id




  #Adding the first (given symptom) and corresponding crossed out


  
  user_graph = {
    "nodes": [],
    "links": []
  }


  root_node = {
    "label": random_symptom,
    "id": 1000000
  }
  our_graph["nodes"].append(root_node)
  user_graph["nodes"].append(root_node)

  #Always "No"
  our_node = {
    "label": "Crossed out",
    "id": our_id_count
  }
  our_graph["nodes"].append(our_node)
  user_graph["nodes"].append(our_node)



  link_to_crossed = {
    "source": 1000000,
    "target": our_id_count,
    "label": "No"
  }

  link_to_target = {
    "source": 1000000,
    "target": 0,
    "label": "Yes"
  }

  our_id_count += 1

  our_graph["links"].append(link_to_crossed)
  our_graph["links"].append(link_to_target)
  user_graph["links"].append(link_to_crossed)
  user_graph["links"].append(link_to_target)



  print("Ideal graph JSON")
  print(our_graph)


  #Id of the next node for user graph
  #For crossed outs we can use the same counter because we are not gonna use it anymore anyways
  user_id_count = 0



  '''
  G = nx.DiGraph()

  # Add nodes with labels
  for node in our_graph['nodes']:
      G.add_node(node['id'], label=node['label'])

  # Add edges with labels
  for link in our_graph['links']:
      G.add_edge(link['source'], link['target'], label=link['label'])

  # Draw the graph
  pos = nx.spring_layout(G, seed=42)
  edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
  print(G.nodes(data=True))
  node_labels = {n: d['label'] for n, d in G.nodes(data=True)}

  nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=800, node_color='skyblue', font_weight='bold')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

  plt.title('Medical Symptoms Graph')
  plt.show()
  '''
  

  #return

  
  got_prognosis = False

  while not got_prognosis:
    inp = input()
    
    query2 = ""

    if isQuestion(inp):
      query2 = f'"{inp}" What is this question about among the following (only tell me the number): '
      
      column_names = df.columns.tolist()
      column_names = column_names[:-1]
      #if debug:
      #  print("All symptoms: ")
      #  print(column_names)

      for i, symptom in enumerate(column_names):
        query2 += f"{i + 1}. {convert_to_title_case(symptom)}; "

    else:
      query2 = f'"{inp}" What is this statement about among the following (only tell me the number): '

      unique_prognoses = df["prognosis"].unique().tolist()
      #if debug:
      #  print("All unique prognoses: ")
      #  print(unique_prognoses)
      #print(i)
      #print(len(column_names))

      for j, this_prognosis in enumerate(unique_prognoses):
        query2 += f"{j + 1}. {this_prognosis}; "

      
      

    if debug:
      print("DEBUG: Query to parse topic in user input: ")
      print(query2)
    
    topic_finding_message = gpt(query2)
    if debug:
      print("DEBUG: Response from gpt about topic: ")
      print(topic_finding_message)




    numbers = re.findall(r'\d+', topic_finding_message)
    
    if len(numbers) == 0:
        print("Sorry, I don't quite get you")
    elif len(numbers) > 1:
        #Sometimes it outputs multiple numbers if the question was too general
        print("What exactly do you mean?")
    else:
        id = int(numbers[0])
        
        if isQuestion(inp):
          #We got symptom
          symptom_queried = column_names[id - 1]
          print("SYSTEM: Symptom queried:", symptom_queried)

          query3 = f'"{inp}" '
          if symptom_queried in symptoms:
            message = 'Patient indeed has this symptom/problem.'
            query3 += message
          else:
            message = "Patient doesn't have this symptom/problem."
            query3 += message

          print("SYSTEM:", message)
            
          query3 += ' Give 1 possible answer patient can give to this question at doctor examination. Length is 1 sentence. Assume doctor has no previous history of patient.'
          if debug:
            print("DEBUG: Query to get answer to user: ")
            print(query3)

          response_to_user = gpt(query3)
          if debug:
            print("DEBUG: Answer suggested by gpt: ")
          print(response_to_user)



          #Graph part is copied from where we add the top part for our_graph
          #Because it is the same

          node = {
            "label": symptom_queried,
            "id": user_id_count
          }
          user_graph["nodes"].append(node)

          node = {
            "label": "Crossed out",
            "id": our_id_count
          }
          user_graph["nodes"].append(our_node)
        


          link_to_crossed = {
            "source": user_id_count,
            "target": our_id_count,
            "label": "No" if symptom_queried in symptoms else "Yes"
          }

          link_to_target = {
            "source": user_id_count,
            "target": user_id_count + 1,
            "label": "Yes" if symptom_queried in symptoms else "No"
          }


          if symptom_queried not in symptoms:
            user_graph["links"].append(link_to_target)
            user_graph["links"].append(link_to_crossed)
          else:
            user_graph["links"].append(link_to_crossed)
            user_graph["links"].append(link_to_target)


          our_id_count += 1
          user_id_count += 1



        else:
          #We got prognosis
          got_prognosis = True

          prognosis_predicted = unique_prognoses[id - 1]
          print("SYSTEM: Predicted prognosis:")
          print(prognosis_predicted)

          if prognosis_predicted == prognosis:
            print("SYSTEM: correct prediction")
          else:
            print("SYSTEM: you failed. Correct prognosis is", prognosis)


          node = {
            "label": prognosis_predicted,
            "id": user_id_count
          }
          user_graph["nodes"].append(node)
        
  print("User graph JSON")
  print(user_graph)      

start()