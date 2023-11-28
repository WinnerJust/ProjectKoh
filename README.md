# Project Koh

### Project description

Project Koh is an AI-based application designed for medical students to practice the diagnostic procedure on simulated patients. 

The app simulates patients with some pathology, corresponding symptoms, and antecedents. The user’s goal is to have a conversation with an imaginary patient and ask relevant questions. Based on the responses of the patient, user has to conclude with a diagnosis.

During the conversation, Project Koh offers hints. By the end of the talk, the application displays the correct result and provides metrics on their diagnosis procedure and suggestions on how to improve it.

### How to run

1. Download /model from Google Drive: [https://drive.google.com/file/d/19uZb7Gl65zzhHdWwgSMc39sSCVo_yVZb/view?usp=sharing](https://drive.google.com/file/d/19uZb7Gl65zzhHdWwgSMc39sSCVo_yVZb/view?usp=sharing)
2. Extract /data/small_train_patients.rar
3. Run /src/project-koh.py

### **Technology**

First, we need to have a model that makes a diagnosis decision given list of symptoms in order to have ground truth diagnosis to compare user’s diagnosis with.

Initial idea was to use Decision Trees because of its nature which resembles a talk of a doctor with a patient, that is the doctor is asking a question, the patient answers it.

Some other models were also considered, among them:

1. **Bayes networks (evidence optimization)**
    
    [https://www.bayesserver.com/examples/demos/ai-doctor](https://www.bayesserver.com/examples/demos/ai-doctor)
    
2. **Deep Q-learning based Rainbow**
    
    [https://github.com/mila-iqia/Casande-RL](https://github.com/mila-iqia/Casande-RL)
    

The problem with those is even though they are obviously more complex and give accurate chain of questions, there is no way to access quality of users’ questions.

Whereas Decision Trees by their nature work trying to minimize entropy with every question. That is how we are going to rate users’ questions and suggest better questions.

### **Dataset**

We are using DDXPlus dataset: [https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374)

This dataset has 49 pathologies, 223 symptoms, and 1025602 (!) rows of training data as mentioned in their paper: [https://arxiv.org/pdf/2205.09148.pdf](https://arxiv.org/pdf/2205.09148.pdf)

Let’s construct a Decision Tree for this dataset.

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled.png)

Initially it is in French. So it was decided to translate it.

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%201.png)

**Adding noise to the dataset**

It was decided to see the distribution of symptoms for every pathology to see if it is diverse enough.

Example: symptom distribution for URTI:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%202.png)

As you can see it is very diverse. But it was decided to try to add some noise to the dataset anyway to see if we can improve accuracy or improve quality/quantity of questions asked.

Trying to add 10 random symptoms for each of 10000 patients in the sampled dataset. Full dataset has 1025602 patients. Accuracy:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%203.png)

Indeed we got more questions. But the problem is that the new questions are irrelevant. For example:

 

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%204.png)

Tree starts to ask about randomly added symptoms in this case about whether patient has pain in his right ring finger which is completely irrelevant.

So, the final dataset is used without any noise.

### Differential diagnosis

Dataset 2 contains differential diagnosis for every patient in addition to ground truth pathology, that is list of pairs: (pathology, probability).

Example:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%205.png)

Predicting differential diagnosis instead of one pathology could have greatly complicated the task, because this requires some other model rather than decision tree. But all alternative models are not able to provide evaluation of users’ questions (as it was mentioned in the section **Technology choosing**).

It was decided to analyze the differential diagnosis data to determine its relevance for our task.

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%206.png)

Here we come to a contraction: most probable pathology has probability of 0.3203 to be actually present (that is to be equal to the ground truth pathology) on average. But in fact in the dataset 2 ground truth pathology is equal to the most probable pathology with the probability of 0.7387.

That means the differential diagnosis is incorrect in this dataset and will not be used in Project Koh.

### Custom Tree building

There were two reasons for developing custom Tree:

1. Dataset 2 contains symptoms that should be asked only after some other symptom was already asked. 
    
    Example: *“Does pain radiate to another location?”* should be asked only after *“Do you feel pain somewhere?”*
    
    Every tree that was built so far decided to put the question about pain radiation as the first question:
    
    ![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%207.png)
    
    The reason behind this is with this question we replace two other questions:
    
    *“Do you feel pain somewhere?”* and *“Does pain radiate to another location?”*.
    
    If the answer to the question on the picture above is “Yes”, then we know that the pain is present and it does not radiate. But asking that question in real diagnosis procedure is inappropriate.
    
    The solution is to allow the custom Tree to remember features that it already split on and condition next features on the previous ones. 
    
2. We need to give hints during diagnosis process
    
    As you know Decision Tree uses information gain concept to find the most optimal questions at every step. We would like to use this concept separately, that is not only to find the most optimal question (to show user as a hint), but to find information gain for the user’s question to determine how good the question is.
    

**Custom Tree evaluation**

Custom Tree was built using parameters found via Grid Search. Accuracy of the custom Tree:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%208.png)

As you can see the accuracy is very close to the accuracy of scikit Tree. And the actual Tree is very similar to the scikit Tree:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%209.png)

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%2010.png)

The slight difference is present because the custom Tree conditions next features on the previous ones.

### **Model**

The main application of neural networks in Project Koh is to analyze user’s input and determine which symptom user is asking about.

This is a sentence classification task which is a common task for pretrained transformers. That is why the model chosen for the Project is GPT (Generative Pre-trained Transformer).

**Demo version**

The demo was build using prompts to GPT3.5 from OpenAI. Let’s say user sent the following message: “Is your body temperature elevated?”. The initial idea was to use the following prompt to GPT:

*“Is your body temperature elevated?” Is this question about: 1. Cough; 2. Fatigue; 3. Fever; 4. Weight loss; … (and all the rest 219 symptoms)*

But some of the answers did not contain symptoms from this list but something else. Due to this, the prompt was refined:

*“Is your body temperature elevated?” What is this question about among the following (only tell me the number): 1. Cough; 2. Fatigue; 3. Fever; 4. Weight loss; … (and all the rest 219 symptoms)*

**Final version**

The final version of the application is using fine-tuned GPT2 from Hugging Face: [https://huggingface.co/gpt2](https://huggingface.co/gpt2)

Dataset for the fine-tuning was collected again using automated prompts to GPT3.5 from OpenAI. Namely the following:

“*Do you have {symptom_name}?” Paraphrize this question. Provide 100 paraphrized questions in a numbered list.*

Fine-tuning was done using modified George Mihaila’s Colab Notebook: [https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb)

Results of the fine-tuning:

![Untitled](Project%20Koh%20e4ca4e0e80ec4b509dbf15651edb73d8/Untitled%2011.png)

Model gives good performance when using the application.

### Final version demonstration

Is available at: [https://github.com/WinnerJust/ProjectKoh](https://github.com/WinnerJust/ProjectKoh)