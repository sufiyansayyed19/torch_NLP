# NLP PyTorch Cheatsheet

## Quick Imports
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, pipeline
```

## Text Preprocessing
```python
import re
text = re.sub(r'[^\w\s]', '', text.lower())  # Clean
tokens = text.split()                          # Tokenize
```

## Embeddings
```python
# PyTorch
embed = nn.Embedding(vocab_size, embed_dim)
vectors = embed(token_ids)  # [batch, seq, dim]

# Pretrained
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('vectors.bin', binary=True)
```

## RNN/LSTM/GRU
```python
rnn = nn.LSTM(input_size, hidden_size, num_layers, 
              batch_first=True, bidirectional=True)
output, (h_n, c_n) = rnn(x)
# output: [batch, seq, hidden*2]  h_n: [layers*2, batch, hidden]
```

## Attention
```python
# Scaled dot-product
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)
```

## HuggingFace
```python
# Pipeline (easiest)
pipe = pipeline("sentiment-analysis")
result = pipe("I love this!")

# Model + Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model(**inputs)
```

## LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

## RAG (Retrieval)
```python
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(documents=docs, ids=ids)
results = collection.query(query_texts=[query], n_results=3)
```

## Metrics
```python
from sklearn.metrics import f1_score, precision_recall_fscore_support
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
```

## Quick Model Choice
| Data Size | Task | Model |
|-----------|------|-------|
| <1K | Classification | TF-IDF + LogReg |
| 1K-10K | Classification | BERT fine-tune |
| Any | Generation | GPT + prompting |
| Private data | QA | RAG |
