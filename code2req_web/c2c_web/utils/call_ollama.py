import multiprocessing
import time
from ollama import chat
import json
from openai import OpenAI
from django.conf import settings

def ollama_call_worker(model, messages, format, tools, options):
    if format:
        return chat(
            model=model,
            messages=messages,
            format=format.model_json_schema(),
            options=options
        ) 
    else:
        if tools:
            return chat(
                model=model,
                messages=messages,
                tools=tools,
                options=options
            )
        else:
            return chat(
                model=model,
                messages=messages,
                options=options
            )

key=settings.OPENAI_KEY
client = OpenAI(
    api_key=key,
)

def call_openai(model, messages, format, tools, options):
    temperature=0
    if "temperature" in options:
        temperature=options["temperature"]
    try:
        if tools:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature
            )
        else:
            if format:
                response = client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=format,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
        return response.choices[0]
    except Exception as e:
        print("LLM call FAILED! ")
        print(e)
        return None
 

def safe_ollama_chat(
    model='qwen2.5-coder',
    messages=None,
    format=None,
    tools=None,
    options={'temperature': 0},
    timeout_seconds=30,
    retries=3,
    backoff_seconds=30
):
    if model=="gpt-4.1" or model=="gpt-5":
        response = call_openai(model, messages, format, tools, options)
        return response
    """
    Safely call Ollama chat() with timeout and retry protection using subprocess.
    """
    for attempt in range(1, retries + 1):
        ctx = multiprocessing.get_context("spawn")  # More robust on Windows
        with ctx.Pool(processes=1) as pool:
            result = pool.apply_async(
                ollama_call_worker, 
                (model, messages, format, tools, options)
            )

            try:
                response = result.get(timeout=timeout_seconds)
                return response
            except multiprocessing.TimeoutError:
                print(f"Ollama call timed out after {timeout_seconds} seconds (attempt {attempt}/{retries})")
                pool.terminate()  # Kill the subprocess immediately
            except Exception as e:
                print(f"Ollama call failed (attempt {attempt}/{retries}): {e}")
                pool.terminate()

        if attempt < retries:
            print(f"Retrying in {backoff_seconds} seconds...")
            time.sleep(backoff_seconds)

    print("All retries exhausted. Ollama call failed.")
    return None


def safe_ollama_chat_tools(
    model='qwen2.5-coder',
    messages=None,
    format=None,
    options={'temperature': 0},
    timeout_seconds=30,
    retries=3,
    backoff_seconds=30,
    tools=None,
    vector_collection=None,
    top_k=5
):
    from c2c_web.utils.tools import get_software_requirement
    make_call=True
    use_format=False
    while make_call==True:
        if use_format==False:
            response = safe_ollama_chat( model=model,messages=messages,options=options,timeout_seconds=timeout_seconds,retries=retries,backoff_seconds=backoff_seconds,tools=tools)
        else:
            response = safe_ollama_chat( model=model,messages=messages,options=options,timeout_seconds=timeout_seconds,retries=retries,backoff_seconds=backoff_seconds,format=format) 
        # print(response)
        if response==None:
            make_call=False
        elif "tool_calls" in response.message or (hasattr(response, 'finish_reason') and response.finish_reason=="tool_calls"):
            messages.append(response.message)
            for call in response.message.tool_calls:
                if call.function.name=="get_software_requirement":
                    try:
                        query = json.loads(call.function.arguments)["query"]
                    except Exception as e:
                        query = call.function.arguments["query"]
                    result = get_software_requirement(query, vector_collection, top_k)
                    if hasattr(call, 'id'):
                        m={'role': 'tool', 'content':  str(result), 'name': call.function.name, 'tool_call_id': call.id}
                    else:
                        m={'role': 'tool', 'content':  str(result), 'name': call.function.name}
                    messages.append(m)
                    use_format=True
                else:
                    print("Unknown function")
        elif isinstance(response.message.content, str) and "{\"name\":" in response.message.content:
            messages.append(response.message)
            try:
                json_start = response.message.content.find('{')
                json_str = response.message.content[json_start:]
                call_obj = json.loads(json_str)
                func_name = call_obj['name']
                if func_name=="get_software_requirement":
                    func_args = call_obj.get('arguments', {})
                    query = func_args["query"]
                    result = get_software_requirement(query, vector_collection, top_k)
                    messages.append({'role': 'tool', 'content':  str(result), 'name': func_name})
                    use_format=True
                else:
                    print("Unknown function")
            except Exception as e:
                print(f"Failed to parse manual JSON tool call: {e}")
        else:
            if use_format==False: # Meaning that it never used tools, and just went ahead and responsed... then format needs to be enforced
                response = safe_ollama_chat( model=model,messages=messages,options=options,timeout_seconds=timeout_seconds,retries=retries,backoff_seconds=backoff_seconds,format=format) 
            make_call=False
    return response
