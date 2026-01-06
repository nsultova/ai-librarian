
##### Disclaimer

This is, for now, a mere POC to understand RAGs better. I want to turn it into sth proper tho, as I could actually use such a tool pretty much.

###### Techstack
* langchain
* FastAPI
* ChromaDB
* SSR w. jinja2

##### Architecture

- **`ingestor`**: Handles reading files (PDF/EPUB), cleaning text, and chunking
- **`storage`**: Manages the Vector Database (ChromaDB).
- **`engine`**: The RAG logic (Embed Query -> Retrieve -> Generate Answer)
- **`web`**: The FastAPI UI layer (Routes + HTML Templates).


```
ai-librarian/
├── data/               # stores vector DB and uploaded files
├── templates/          # HTML files (Jinja2)
├── static/             # .scss and .css files
├── src/
│   ├── __init__.py
│   ├── config.py       # general config params
│   ├── ingest.py       # loaders & splitters
│   ├── vector.py       # database interface
│   ├── rag.py          # actual rag/llm
│   └── app.py          # webserver
└── requirements.txt
```

#### For Archlinux

Setup uv (a nice tool to handle .venvs):
`uv venv ai-librarian-env`
`source ./ai-librarian-env/bin/activate`
`uv pip install -r reqirements.txt`

**Note:** Install ollama systemwide
`sudo pacman -S ollama`

* make sure ollama is enabled `sudo systemctl start ollama`
* grab the model `ollama pull llama3.2`
* start Ollama: `ollama run llama3.2` (or whichever model you configured)
* run the server: `python -m src.app`
* go to `http://localhost:8000`

#### TODO
* include `metadata.py`
* refine metadata
* ingest multiple formats
* add multimodality
* try with quicker models
* improve UI
* build proper design?
* make actually useful


##### MODELS USED

Embedding: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
LLM: https://huggingface.co/meta-llama/Llama-3.2-1B