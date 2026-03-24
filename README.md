
##### Disclaimer

This is, for now, a mere proof-of-concept tinkering with RAG building my own personal librarian - a delightful idea for a notorious but equally forgetful bookworm :D
I want to turn it into sth proper tho, as I could actually use such a tool pretty much.

It is not production-ready (yet). 

###### Techstack
* langchain
* FastAPI
* ChromaDB
* SSR w. jinja2

**Models**
* Embedding: https://huggingface.co/intfloat/e5-small-v2
* LLM: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

**!!!** If you change the embedding-model you NEED to reingest everything

##### Architecture

```
ai-librarian/
├── data/               # stores vector DB and uploaded files
├── templates/          # HTML files (Jinja2)
├── static/             # .scss and .css files
├── src/
│   ├── __init__.py
│   ├── app.py          # webserver
│   ├── config.py       # general config params
│   ├── ingest.py       # loaders & splitters
│   ├── vector.py       # database interface
│   ├── rag.py          # actual rag/llm
|   ├── metadata.py     # preprocessing documents
|   ├── reset_db.py     # cleanup script for ChromaDB
|   ├── utils.py        # shared text-processing utilities
│   └── test_suite.py   # various tests for different modules  
└── requirements.txt
```

**Dataflow**
`upload → ingest → embed → store → retrieve → generate`

 ![Dataflow](/img/basic_RAG_white_background.png)

#### For Archlinux

Environments are managed via uv

```
source .venv/bin/activate
uv sync
```

**Note:** Install ollama systemwide
`sudo pacman -S ollama`

* make sure ollama is enabled `sudo systemctl start ollama`
* grab the model `ollama pull llama3.2`
* start Ollama: `ollama run llama3.2` (or whichever model you configured)
* run the server: `python -m src.app`
* go to `http://localhost:8000`



#### RUN

Always from the project root (`ai-librarian/`), never from inside `src/`:

```
python -m src.app          # run the server
python -m src.test_models  # run all tests
python -m src.test_models rag  # run one test
```


#### Various tests

`python -m src.test_models` - run all tests
`python -m src.test_models rag`  - run one test


`curl http://localhost:8000/library` — should show your books with chapters populated
`curl "http://localhost:8000/debug/search?q=what+happens+in+the+first+chapter"` — inspect whether retrieved chunks are semantically relevant and whether metadata looks clean

**Test UI-DEMO**
Determine if it's your cache that must be cleaned or sth more serious

`python -m http.server 8080`

navigate to
`http://127.0.0.1:8080/templates/preview.html`



#### TODO
* include `metadata.py`
* implement metadata extraction -done
* refine metadata
* add tagging system
* ingest multiple formats - partially done
* try with quicker models - done
* improve retrieval itself
* improve UI
    * Feedback on ingestion
* build proper design?
* make actually useful
* list/show ingested books - done
* backup database 
* add multimodality
* add caching where necessary
* add pagination(?)
* Real-time Progress Streaming (Server-Sent Events)
    * ..if I ever do sthsth-scaling/multi-user/big files
* queries all metadata from DB directly (no LLM)
* collects unique chapters per book into a sorted list (no LLM)
* add summarization of each book to metadata
* add summarization of chapter to be dispalyed along choice
* remove redundancy in UI (filters)
* Do reasonable delete fuction for chromaDB entries and/or full db delete
