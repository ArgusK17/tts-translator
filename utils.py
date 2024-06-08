from openai import OpenAI
import json
import torch

# def prepare_embedding(client, history):
#   num_rows = len(history)
#   embbeding_example = get_embedding(client, history[0]['English'])
#   embedding_history = torch.zeros(num_rows, embbeding_example.shape[0])

#   for row_idx in range(num_rows):
#     input = history[row_idx]['English']
#     embedding = get_embedding(client, input)
#     embedding_history[row_idx] = embedding

#   torch.save(embedding_history, '/content/tts_translator/datasets/translation_history_embedding.pt')

def remove_repeat_words(word_list):
  seen = set()
  unique_list = []
  for item in word_list:
      if item['English'] not in seen:
          unique_list.append(item)
          seen.add(item['English'])
  return unique_list

class Translator():

  def __init__(self, client, glossary_path = '/content/tts_translator/datasets/translation_glossary.json',
         history_path = '/content/tts_translator/datasets/translation_history.json',
         embedding_path = '/content/tts_translator/datasets/translation_history_embedding.pt'):
    self.client = client
    with open(glossary_path, 'r', encoding='utf-8') as file_g:
      self.glossary = json.load(file_g)
    with open(history_path, 'r') as file_h:
      self.history = json.load(file_h)
    self.embedding_history = torch.load(embedding_path)
    
  def get_embedding(self, text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = self.client.embeddings.create(input=[text], model=model).data[0].embedding
    return torch.tensor(embedding)
  
  def get_similar_paragraphs(self, input, num_paras=3):
    embedding = self.get_embedding(input)
    score = (self.embedding_history*embedding.unsqueeze(dim=0)).sum(dim=1)
    _, idx_tensor = torch.topk(score, num_paras)
    idx_list = idx_tensor.tolist()
    similar_paragraphs = [{'English':self.history[row_idx]['English'], "Chinese":self.history[row_idx]['Chinese']} for row_idx in idx_list]
    return similar_paragraphs
  
  def find_matches(self, words):
    matched_pairs = []
    unmatched_words = []

    for word in words:
        # Filter glossary for entries containing the word (case insensitive)
        matches = [entry for entry in self.glossary if word.lower() in entry['English'].lower()]
        
        if not matches:
            unmatched_words.append(word)
        else:
          if not (matches in matched_pairs):
            matched_pairs.extend(matches)  # Add found matches to the list

    return remove_repeat_words(matched_pairs), list(set(unmatched_words))

  def retriveSpecialWords(self, input, model = "gpt-3.5-turbo"):

    searchPromptTemplate = f'''
    Identify special words in the input sentence, including character names, place names, and proper nouns (which usually begin with capital letters). If there are none, return 'NONE'.

    ### Example
    Input: Akemi Homura is a magical girl who lives in Mitakihara City.
    Output: Akemi Homura, Mitakihara City

    ### New Sentence
    Input: {input}
    Output:
    '''

    response = self.client.chat.completions.create(
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

  def augmentSpecialWords(self, input):
      # Get the matching English-Chinese pairs
      word_list = self.retriveSpecialWords(input)
      matched_pairs, unmatched_words = self.find_matches(word_list)
      return matched_pairs, unmatched_words
  
  def translate(self, input:str):
    matched_pairs, unmatched_words = self.augmentSpecialWords(input)
    word_prompt = '### Special words\n'
    for augWord in matched_pairs:
      word_prompt += f'English: {augWord["English"]}\nChinese: {augWord["Chinese"]}\n'
    word_prompt += '\n'
    
    if not unmatched_words == []:
      print(f'{unmatched_words} not found in the glossary')

    similar_paragraphs = self.get_similar_paragraphs(input)
    icl_prompt = '### Transltion Examples\n'
    for para in similar_paragraphs:
      icl_prompt += f'English: {para["English"]}\nChinese: {para["Chinese"]}\n'
    icl_prompt += '\n'

    system_prompt = 'You are an expert novel translator.'
    input_prompt = 'I want you to translate some English paragraphs into Chinese. I provide some reference materials for you.\n\n'+word_prompt+icl_prompt+f'---\nNow translate the following English paragraph into Chinese:\nEnglish: {input}\nChinese: '

    model = "gpt-4o-2024-05-13"
    response = self.client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt},
    ],
    temperature=1,
    )

    return response.choices[0].message.content