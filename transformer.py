
from pathlib import Path
import torch
import torch.nn as nn
import math

device = 'mps'

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
    """
    vocab_size - the size of our vocabulary
    d_model - the dimension of our embeddings and the input dimension for our model
    """
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model) # scale embeddings by square root of d_model


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    positional_embeddings = torch.zeros(seq_len, d_model)
    positional_sequence_vector = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    positional_model_vector = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    positional_embeddings[:, 0::2] = torch.sin(positional_sequence_vector * positional_model_vector)
    positional_embeddings[:, 1::2] = torch.cos(positional_sequence_vector * positional_model_vector)
    positional_embeddings = positional_embeddings.unsqueeze(0)

    self.register_buffer('positional_embeddings', positional_embeddings)

  def forward(self, x):
    x = x + (self.positional_embeddings[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)
  

class LayerNormalization(nn.Module):
  def __init__(self, features: int, epsilon:float=10**-6) -> None:
    super().__init__()
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    standard_deviation = x.std(dim = -1, keepdim = True)
    return self.gamma * (x - mean) / (standard_deviation + self.epsilon) + self.beta
  

class ResidualConnection(nn.Module):
  def __init__(self, features: int, dropout: float = 0.1) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.layernorm = LayerNormalization(features)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.layernorm(x)))


class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
    """
    d_model - dimension of model
    d_ff - dimension of feed forward network
    dropout - regularization measure
    """
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model is not divisible by h"

    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)

    self.w_o = nn.Linear(d_model, d_model, bias=False)

    self.dropout = nn.Dropout(dropout)


def attention(query, key, value, mask, d_k, dropout: nn.Dropout = None):
  attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

  if mask is not None:
    attention_scores.masked_fill_(mask == 0, -1e9)

  attention_scores = attention_scores.softmax(dim=-1)

  if dropout is not None:
    attention_scores = dropout(attention_scores)

  return (attention_scores @ value), attention_scores


def forward(self, query, key, value, mask):
  query = self.w_q(query)
  key = self.w_k(key)
  value = self.w_v(value)

  query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
  key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
  value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

  x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

  x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

  return self.w_o(x)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model is not divisible by h"

    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)

    self.w_o = nn.Linear(d_model, d_model, bias=False)

    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout = None):
    d_k = query.shape[-1]

    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)

    attention_scores = attention_scores.softmax(dim=-1)

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return (attention_scores @ value), attention_scores

  def forward(self, query, key, value, mask):
    query = self.w_q(query)
    key = self.w_k(key)
    value = self.w_v(value)

    query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

    x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

    return self.w_o(x)
  

class EncoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

  def forward(self, x, input_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, input_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x
  

class EncoderStack(nn.Module):
  def __init__(self, features: int, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization(features)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  

class DecoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

  def forward(self, x, encoder_output, input_mask, target_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, input_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x
  

class DecoderStack(nn.Module):
  def __init__(self, features: int, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization(features)

  def forward(self, x, encoder_output, input_mask, target_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, input_mask, target_mask)
    return self.norm(x)
  

class LinearProjectionLayer(nn.Module):
  def __init__(self, d_model, vocab_size) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x) -> None:
    return self.proj(x)
  


class Transformer(nn.Module):
  def __init__(self, encoder: EncoderBlock, decoder: DecoderBlock, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: LinearProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

  def project(self, x):
    return self.projection_layer(x)
  


def build_transformer(input_vocab_size: int, target_vocab_size: int, input_seq_len: int, target_seq_len: int, d_model: int=512, N: int=6, num_heads: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
  input_embeddings = InputEmbeddings(d_model, input_vocab_size)
  target_embeddings = InputEmbeddings(d_model, target_vocab_size)

  input_position = PositionalEncoding(d_model, input_seq_len, dropout)
  target_position = PositionalEncoding(d_model, target_seq_len, dropout)

  encoder_blocks = []

  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  decoder_blocks = []

  for _ in range(N):
    decoder_self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
    decoder_cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)

  encoder_stack = EncoderStack(d_model, nn.ModuleList(encoder_blocks))
  decoder_stack = DecoderStack(d_model, nn.ModuleList(decoder_blocks))

  linear_projection_layer = LinearProjectionLayer(d_model, target_vocab_size)

  transformer = Transformer(encoder_stack, decoder_stack, input_embeddings, target_embeddings, input_position, target_position, linear_projection_layer)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer



from torch.utils.data import Dataset

class BilingualDataset(Dataset):
  def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
    super().__init__()
    self.seq_len = seq_len

    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang

    self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    src_target_pair = self.ds[idx]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.tgt_lang]

    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        raise ValueError("Sentence is too long")

    encoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    decoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    label = torch.cat(
        [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
        "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
        "label": label,
        "src_text": src_text,
        "tgt_text": tgt_text,
    }

def causal_mask(size):
  mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
  return mask == 0


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_tokenizer(config, ds, lang):
  tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
  tokenizer.pre_tokenizer = Whitespace()
  trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
  tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
  return tokenizer


from torch.utils.data import DataLoader, random_split

def get_ds(config):
  # It only has the train split, so we divide it overselves
  ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

  # Build tokenizers
  tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

  # Keep 90% for training, 10% for validation
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

  # Find the maximum length of each sentence in the source and target sentence
  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')


  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
  model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
  return model


def get_weights_file_path(config, epoch: str):
  model_folder = f"{config['datasource']}_{config['model_folder']}"
  model_filename = f"{config['model_basename']}{epoch}.pt"
  return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
  model_folder = f"{config['datasource']}_{config['model_folder']}"
  model_filename = f"{config['model_basename']}*"
  weights_files = list(Path(model_folder).glob(model_filename))
  if len(weights_files) == 0:
      return None
  weights_files.sort()
  return str(weights_files[-1])


import warnings
from tqdm import tqdm
import os

def train_model(config):
  # Define the device
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
  print("Using device:", device)
  if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
  else:
    print("Please ensure you're in a GPU enabled Colab Notebook instance.")
  device = torch.device(device)

  # Make sure the weights folder exists
  Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  initial_epoch = 0
  global_step = 0

  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']):
    torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:
      encoder_input = batch['encoder_input'].to(device)
      decoder_input = batch['decoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)
      decoder_mask = batch['decoder_mask'].to(device)

      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
      proj_output = model.project(decoder_output)

      label = batch['label'].to(device)

      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

      loss.backward()

      optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      global_step += 1

    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'global_step': global_step
    }, model_filename)



config = {
  "batch_size": 16,
  "num_epochs": 10,
  "lr": 10**-4,
  "seq_len": 350,
  "d_model": 512,
  "datasource": 'opus_books',
  "lang_src": "en",
  "lang_tgt": "it",
  "model_folder": "weights",
  "model_basename": "encoder_decoder_model_"
}


# decode to produce target/output words
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id("[SOS]") # start of sentence
  eos_idx = tokenizer_tgt.token_to_id("[EOS]") # end of sentence

  # encode the source sentence 
  encoder_output = model.encode(source, source_mask)

  # create a tensor to hold decoded words
  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

  while True:
    # break if we reach max_len
    if decoder_input.size(1) >= max_len:
      break

    # create a mask to prevent the decoder from attending to future words
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    # get the output of the decoder
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    # predict the probabilities of each word
    prob = model.project(out[:, -1])

    # get the word with the highest probability
    _, next_word = torch.max(prob, dim=-1)

    # append the word to the decoder input
    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

    # break if we predict the end of sentence token
    if next_word == eos_idx:
      break

  # convert the decoded words to tokens
  return decoder_input.squeeze(0)



def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=1):
  model.eval()
  count = 0
  
  source_texts = []
  expected = []
  predicted = []

  console_width = 80

  with torch.no_grad():
    for batch in val_dataloader:
      count += 1
      encoder_input = batch['encoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)

      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

      # get the decoded tokens
      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

      source_text = batch['src_text'][0]
      target_text = batch['tgt_text'][0]

      # convert the tokens to text
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      source_texts.append(source_text)
      expected.append(target_text)
      predicted.append(model_out_text)

      print_msg('-' * console_width)
      print_msg(f"Source: {source_text}") 
      print_msg(f"Expected: {target_text}") 
      print_msg(f"Predicted: {model_out_text}")

      if count >= num_examples:
        break


def predict_model(config):
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
  print("Using device:", device)
  if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
  else:
    print("Please ensure you're in a GPU enabled Colab Notebook instance.")
  device = torch.device(device)
  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  model_filename = "/Users/mitultiwari/projects/course/LLME-The-Fundamentals-Cohort-2/models/opus_books_weights/encoder_decoder_model_09.pt"
  checkpoint = torch.load(model_filename)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  # run inference on part of validation dataset
  run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)


if __name__ == '__main__':
  # to train model 
  # train_model(config)
  predict_model(config)
  

