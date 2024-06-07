import numpy as np
import torch

# Function to retrive special word with LLM
def retriveSpecialWords(client, input, model = "gpt-3.5-turbo"):

    searchPromptTemplate = f'''
    Identify special words in the input sentence, including character names, place names, and proper nouns (which usually begin with capital letters). If there are none, return 'NONE'.

    ### Example
    Input: Akemi Homura is a magical girl who lives in Mitakihara City.
    Output: Akemi Homura, Mitakihara City

    ### New Sentence
    Input: {input}
    Output:
    '''

    response = client.chat.completions.create(
    model=model,
    messages=[
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": searchPromptTemplate},
    ],
    temperature=0,
    )

    string = response.choices[0].message.content
    if string == 'NONE':
        word_list = []
    else:
        word_list = [item.strip() for item in string.split(',')]

    return word_list

def remove_repeat_words(word_list):
  seen = set()
  unique_list = []
  for item in word_list:
      if item['English'] not in seen:
          unique_list.append(item)
          seen.add(item['English'])
  return unique_list

# Function to find matches
def find_matches(words, glossary):
    matched_pairs = []
    unmatched_words = []

    for word in words:
        # Filter glossary for entries containing the word (case insensitive)
        matches = [entry for entry in glossary if word.lower() in entry['English'].lower()]
        
        if not matches:
            unmatched_words.append(word)
        else:
          if not (matches in matched_pairs):
            matched_pairs.extend(matches)  # Add found matches to the list

    return remove_repeat_words(matched_pairs), list(set(unmatched_words))

def augmentSpecialWords(client, input, glossary):
    # Get the matching English-Chinese pairs
    word_list = retriveSpecialWords(client, input)
    matched_pairs, unmatched_words = find_matches(word_list, glossary)
    return matched_pairs, unmatched_words

def get_embedding(client, text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return torch.tensor(embedding)

def get_similar_paragraphs(client, input, embedding_history, history, num_paras=3):
  embedding = get_embedding(client, input)
  score = (embedding_history*embedding.unsqueeze(dim=0)).sum(dim=1)
  _, idx_tensor = torch.topk(score, num_paras)
  idx_list = idx_tensor.tolist()
  similar_paragraphs = [{'English':history[row_idx]['English'], "Chinese":history[row_idx]['Chinese']} for row_idx in idx_list]
  return similar_paragraphs