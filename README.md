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

   Repository code-to-feature prompts are packed under a separate input-token budget. The default configuration uses a 4,096-token estimated input ceiling, an 8,192-token context, a 1,024-token output reserve, and a 512-token safety margin. These values can be overridden with the `REPOSITORY_FEATURE_*` environment variables in `docker-compose.yml`; `REPOSITORY_FEATURE_INPUT_TOKEN_LIMIT` is an input budget, not the requirement-generation `max_tokens` setting.

   The fallback counter uses GPT-4 `tiktoken` and is only an estimate for Qwen. A short local probe with `qwen2.5-coder:32b` counted 443 prompt tokens in Ollama versus a 434-token raw application estimate before its additional framing allowance; this is one calibration sample, not a general error bound. The application adds a fixed framing allowance, and the remaining safety margin is configurable. Check logged estimates against Ollama's `prompt_eval_count` on representative prompts for each model/template. For model-specific content tokenization, provide a local `tokenizer.json` matching the exact Ollama model tag and configure the web service with `REPOSITORY_FEATURE_TOKENIZER_JSON` (container path) and `REPOSITORY_FEATURE_TOKENIZER_MODEL` (exact selected model tag). Mount that file read-only into the web container, for example:

   ```yaml
   services:
      web:
         volumes:
            - ./tokenizers/qwen2.5-coder-32b/tokenizer.json:/opt/code2req/tokenizer.json:ro
         environment:
            - REPOSITORY_FEATURE_TOKENIZER_JSON=/opt/code2req/tokenizer.json
            - REPOSITORY_FEATURE_TOKENIZER_MODEL=qwen2.5-coder:32b
            - REPOSITORY_FEATURE_REQUIRE_MODEL_TOKENIZER=true
   ```

   Use a tokenizer file/version corresponding to the served model. The local tokenizer counts message text; Ollama's exact chat template and framing are not reproduced, so a safety margin is still required. Feature-generation logs label the estimator and report Ollama `prompt_eval_count`, duration, and stop reason when available. Start with strict mode disabled until the matching tokenizer is installed; strict mode refuses to fall back to GPT tokenization.

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
