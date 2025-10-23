# Code2Req

**Code2Req** is a framework for **generating requirements from source code** using large language models (LLMs) and a connected knowledge graph database.

---

##  Run the Application

You can run Code2Req either with **Ollama (local models)** and using the **OpenAI API**.

---

### 1. Prerequisites

Before running the application, make sure you have the following installed:

- **Docker** and **Docker Compose**
- **Ollama** 

---

### 2. Configure LLMs

#### Set up Ollama for local models

1. Install [Ollama](https://ollama.ai/).  
2. Download the required models:

   ```bash
   ollama pull mxbai-embed-large
   ollama pull qwen2.5-coder:32b
   ```

---

#### Configure OpenAI API (optional)

If you prefer to use OpenAI models instead of local ones:

1. Get an API key from [OpenAI](https://platform.openai.com/).  
2. Open the `settings.py` file in your Code2Req project.  
3. Set your key as:

   ```python
   OPENAI_KEY = "your-api-key-here"
   ```

---

### 3. Run in Development Mode with Docker

#### Step 1 - Configure your paths in docker-compose.yml

Open docker-compose.yml

The `graphdb_data` volume maps to a local folder so data persists between run, change it to a suitable location on your system

All code is mounted as a **Docker volume**, so changes to local files automatically reflect inside the container.

#### Step 2 — Build the container

From the project root:

```bash
docker compose build
```

#### Step 3 — Start the application

Then run:

```bash
docker compose up
```

Once the containers are up, the web application will be available at:

[http://localhost:8000](http://localhost:8000)

GraphDB workbench will be available at:

[http://localhost:7200](http://localhost:7200)

---
