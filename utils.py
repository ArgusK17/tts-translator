from tts_translator.identify import get_embedding
import torch

def prepare_embedding(client, history):
  num_rows = len(history)
  embbeding_example = get_embedding(client, history[0]['English'])
  embedding_history = torch.zeros(num_rows, embbeding_example.shape[0])

  for row_idx in range(num_rows):
    input = history[row_idx]['English']
    embedding = get_embedding(client, input)
    embedding_history[row_idx] = embedding

  torch.save(embedding_history, '/content/tts_translator/datasets/translation_history_embedding.pt')