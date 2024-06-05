import pandas as pd

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

# Function to find matches
def find_matches(words, glossary):
    matched_pairs = []
    for word in words:
        matches = glossary[glossary['English'].str.contains(word, case=False, na=False)]
        for _, match in matches.iterrows():
            matched_pairs.append((match['English'], match['Chinese']))
    return matched_pairs

def augmentSpecialWords(client, input):
    glossary_path = '/content/tts_translator/datasets/translation_glossary.csv'
    glossary = pd.read_csv(glossary_path, skiprows=1, usecols=[0, 1], header=None)
    glossary.columns = ['English', 'Chinese']

    # Get the matching English-Chinese pairs
    word_list = retriveSpecialWords(client, input)
    matches = find_matches(word_list, glossary)
    return matches