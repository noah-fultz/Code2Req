from c2c_web.models import EvaluationFile, EvaluationUnit 
from ollama import chat
from c2c_web.utils.prompts import tools_requirements, get_generate_req_from_feature_sys_prompt, get_generate_req_from_code_sys_prompt, get_generate_req_from_feature_user_prompt, get_generate_req_from_code_user_prompt
from c2c_web.utils.utils import get_all_from_vectordb, delete_collection, get_or_create_collection, embed, add_to_vectordb
import csv
from typing import Dict
from rdflib import  Namespace, URIRef, RDF
from pathlib import Path
from c2c_web.utils.prompts import RequirementList
from c2c_web.utils.call_ollama import safe_ollama_chat, safe_ollama_chat_tools
import tiktoken
import chromadb

def get_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-5")
    return len(encoding.encode(text))

def extract_name(value):
    name = value.split("#")[-1]
    return name

def get_artifact_ids(collection_name: str) -> Dict[str, str]:
    """Fetches all documents from a given ChromaDB vector collection."""
    results = get_all_from_vectordb(collection_name)
    feature_names_to_ids={}
    for i, meta in enumerate(results["metadatas"]):
        feature_name=extract_name(meta["source"])
        feature_names_to_ids[feature_name]=meta["source"]
    return feature_names_to_ids

def read_chunks_from_file(filepath, feature_name_ids):
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("ID") and row.get("Description"):
                if row.get("CodeArtifacts"):
                    code_artifacts=set()
                    chunks.append({
                    "id": row["ID"].strip(),
                    "description": row["Description"].strip(),
                    "code_artifacts": list(code_artifacts),
                })
                else:
                    chunks.append({
                        "id": row["ID"].strip(),
                        "description": row["Description"].strip(),
                        "code_artifacts":[]
                    })
    return chunks

def get_level1_features(graph,EX):
    all_features={}
    root=""
    for s, p, o in graph.triples((None, EX.isFeatureFor, None)):
        s_str, o_str = str(s), str(o)
        if o_str not in all_features:
            desc = graph.value(subject=o, predicate=EX.description)
            name = graph.value(subject=o, predicate=EX.name)
            level = graph.value(subject=s, predicate=EX.level)
            if str(level)=="0":
                all_features[o_str]=str(name)+": "+str(desc)
                if root=="":
                    desc = graph.value(subject=s, predicate=EX.description)
                    name = graph.value(subject=s, predicate=EX.name)
                    root=name+": "+desc
    return all_features, root

def get_all_features(graph, EX):
    all_features={}
    root=""
    for s, p, o in graph.triples((None, EX.isFeatureFor, None)):
        level = graph.value(subject=s, predicate=EX.level)
        if str(level)=="0":
            desc = graph.value(subject=s, predicate=EX.description)
            name = graph.value(subject=s, predicate=EX.name)
            all_features["Feature_hierarchy"]={"Name":str(name), "Description":str(desc), "IsFeatureFor":[]}
            root=s
    
    root_children=[]
    for root_child in graph.objects(subject=root, predicate=EX.isFeatureFor):
        desc = graph.value(subject=root_child, predicate=EX.description)
        name = graph.value(subject=root_child, predicate=EX.name)
        next_children=[]
        for next_child in graph.objects(subject=root_child, predicate=EX.isFeatureFor):
            if (next_child, RDF.type, EX.Feature) in graph:
                desc_child = graph.value(subject=next_child, predicate=EX.description)
                name_child = graph.value(subject=next_child, predicate=EX.name)
                next_children.append({"Name":str(name_child), "Description":str(desc_child), "IsFeatureFor":[]})
        root_children.append({"Name":str(name), "Description":str(desc), "IsFeatureFor":next_children})
    all_features["Feature_hierarchy"]["IsFeatureFor"]=root_children
    print(all_features)

    return all_features

def get_all_level_2_features(graph, EX):
    all_features=[]
    added_features=[]
    for s, p, o in graph.triples((None, EX.isFeatureFor, None)):
        level = graph.value(subject=o, predicate=EX.level)
        if str(level)=="2":
            if o not in added_features:
                desc = graph.value(subject=o, predicate=EX.description)
                name = graph.value(subject=o, predicate=EX.name)
                all_features.append({"Name":str(name), "Description":str(desc)})
                added_features.append(o)

    print(all_features)
    print(len(all_features))

    return all_features


def get_feature_classes_for_level1(graph, EX, feature_id):
    if not isinstance(feature_id, URIRef):
        feature_uri = URIRef(feature_id if feature_id.startswith("http") else str(EX[feature_id]))
    else:
        feature_uri = feature_id

    class_artifacts = set()

    # Step 1: Find child features: feature_uri --EX.isFeatureFor--> child_feature
    for child_feature in graph.objects(subject=feature_uri, predicate=EX.isFeatureFor):
        
        # Step 2: For each child_feature, find associated code artifacts: child_feature --EX.isFeatureFor--> artifact
        for artifact in graph.objects(subject=child_feature, predicate=EX.isFeatureFor):

            # Step 3: Check if artifact is of type EX.Class
            if (artifact, RDF.type, EX.Class) in graph:
                class_artifacts.add(str(artifact))
    return list(class_artifacts)

def get_child_features(graph,EX, parent):
    all_features=[]
    for s, p, o in graph.triples((None, EX.isFeatureFor, None)):
        s_str, o_str = str(s), str(o)
        if s_str==parent:
            if (o, RDF.type, EX.Feature) in graph:
                desc = graph.value(subject=o, predicate=EX.description)
                name = graph.value(subject=o, predicate=EX.name)
                all_features.append(str(name)+": "+str(desc))
    return all_features

def get_reqs_from_feature(model, top_k, max_tokens, graph, EX, example_1, example_2, example_3, format, vector_collection_requirements, use_rag):
    one_feature_at_the_time=True
    if max_tokens>=100000:
        one_feature_at_the_time=False
    reqs=[]
    related_classes=[]
    if one_feature_at_the_time==True:
        features, root = get_level1_features(graph, EX)
        print("Working with num FEATURES: "+str(len(features)))
        for i, f in enumerate(features):
            messages=[{'role': 'system', 'content':get_generate_req_from_feature_sys_prompt(example_1, example_2, example_3, format, use_rag)}]
            messages.append({'role': 'user', 'content': "Here is a high-level feature: "+str(features[f])})
            children=get_child_features(graph, EX, f)
            if children!=[]:
                messages.append({'role': 'user', 'content': "Here are its underlying lower-level features: "+str(children)})
            messages.append({'role': 'user', 'content': get_generate_req_from_feature_user_prompt(use_rag)})
            text=""
            for m in messages:
                text+=m["content"]
            print("Sending tokens: "+str(get_tokens(text)))
            if use_rag==True:
                response = safe_ollama_chat_tools(model=model, top_k=top_k,messages=messages,format=RequirementList, tools=tools_requirements, vector_collection=vector_collection_requirements)
            else:
                response = safe_ollama_chat(model=model,messages=messages,format=RequirementList)
            if response!=None:
                abstraction = RequirementList.model_validate_json(response.message.content)
                for abstracted_req in abstraction.requirements:
                    reqs.append(abstracted_req)
    else:
        all_features=get_all_features(graph, EX)
        messages=[{'role': 'system', 'content':get_generate_req_from_feature_sys_prompt(example_1, example_2, example_3, format, use_rag)}]
        messages.append({'role': 'user', 'content': "Here are the hierarchical features for the application: "+str(all_features)})
        messages.append({'role': 'user', 'content': get_generate_req_from_feature_user_prompt(use_rag)})
        text=""
        for m in messages:
            text+=m["content"]
        print("Sending tokens: "+str(get_tokens(text)))
        if use_rag==True:
            response = safe_ollama_chat_tools(model=model, top_k=top_k,messages=messages,format=RequirementList, tools=tools_requirements, vector_collection=vector_collection_requirements)
        else:
            response = safe_ollama_chat(model=model,messages=messages,format=RequirementList)
        if response!=None:
            abstraction = RequirementList.model_validate_json(response.message.content)
            for abstracted_req in abstraction.requirements:
                reqs.append(abstracted_req)
        
    return reqs, related_classes

def safe_read_text(file_path):
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding='utf-8-sig')  # handles BOM
        except UnicodeDecodeError:
            return file_path.read_text(encoding='ISO-8859-1')  # or 'cp1252'

def read_code(dir_path, code_list_per_file):
    for entry in dir_path.iterdir():
        if entry.name == "__MACOSX":
            continue 
        if entry.is_dir():
            code_list_per_file = read_code(entry, code_list_per_file)
        elif entry.suffix == ".py" or entry.suffix == ".java":
            code = safe_read_text(entry)
            code_list_per_file[str(entry)]=code
    return code_list_per_file

def all_in_context(model, top_k, REPO_ROOT_PATH, example_1, example_2, example_3, format, use_rag, vector_collection_requirements, MAX_TOKENS = 400000):
    code_list_per_file = {}
    code_list = read_code(REPO_ROOT_PATH, code_list_per_file)
    total_tokens=0
    for c in code_list:
        total_tokens+=get_tokens(code_list[c])
    print("Total code tokens: "+str(total_tokens))
    files = list(code_list.keys())
    responses = []

    # Initialize messages with system prompt
    system_msg = {
        'role': 'system',
        'content': get_generate_req_from_code_sys_prompt(example_1, example_2, example_3, format, use_rag)
    }
    system_tokens = get_tokens(system_msg['content'])
    query_tokens = get_tokens(get_generate_req_from_code_user_prompt(use_rag))

    batch_messages = [system_msg]
    token_count = system_tokens

    sent_batches=0
    for idx, filename in enumerate(files):
        code_msg = {'role': 'user', 'content': f"{filename}:\n{code_list[filename]}"}
        code_tokens = get_tokens(code_msg['content'])
        if token_count + code_tokens + query_tokens > MAX_TOKENS:
            # Add query to current batch
            batch_messages.append({'role': 'user', 'content': get_generate_req_from_code_user_prompt(use_rag)})
            print(f"Sending dynamic batch with ~{token_count} tokens...")
            sent_batches+=1
            if use_rag==True:
                response = safe_ollama_chat_tools(model=model, top_k=top_k,messages=batch_messages,format=RequirementList, tools=tools_requirements, vector_collection=vector_collection_requirements)
            else:
                response = safe_ollama_chat(model=model,messages=batch_messages,format=RequirementList)
            if response!=None:
                abstraction = RequirementList.model_validate_json(response.message.content)
                for abstracted_req in abstraction.requirements:
                    responses.append(abstracted_req.description)
                batch_messages = [system_msg]
                token_count = system_tokens
        # Add file message to current batch
        batch_messages.append(code_msg)
        token_count += code_tokens

    # Handle remaining batch (if any files left unprocessed)
    if len(batch_messages) > 1:
        batch_messages.append({'role': 'user', 'content': get_generate_req_from_code_user_prompt(use_rag)})
        print(f"Sending final batch with ~{token_count} tokens...")
        sent_batches+=1
        if use_rag==True:
            response = safe_ollama_chat_tools(model=model, top_k=top_k,messages=batch_messages,format=RequirementList, tools=tools_requirements, vector_collection=vector_collection_requirements)
        else:
            response = safe_ollama_chat(model=model,messages=batch_messages,format=RequirementList)
        if response!=None:
            abstraction = RequirementList.model_validate_json(response.message.content)
            for abstracted_req in abstraction.requirements:
                responses.append(abstracted_req.description)
    print("Sent batches: "+str(sent_batches))
    return responses

def process_eval_files(project, graph, ex, repo_root, vector_collection_requirements, vector_path, model, top_k, max_tokens, rag, use_feature, use_code, use_rag=False):
    EX = Namespace(ex + "#")
    REPO_ROOT_PATH = Path(repo_root).resolve()
    example_1=""
    example_2=""
    example_3=""
    format=""
    feature_name_ids = {} 
    for eval_file in project.evaluation_files.all():
        if eval_file.name!="Code_Features" and eval_file.name!="Req_from_code" and eval_file.name!="Code_Features_RAG" and eval_file.name!="Req_from_code_RAG":
            chunks = read_chunks_from_file(eval_file.path, feature_name_ids)
            client = chromadb.PersistentClient(path=vector_path)
            delete_collection(client, vector_collection_requirements)
            collection=get_or_create_collection(client, vector_collection_requirements)
            print("Made collection")
            for chunk in chunks:
                identifier = chunk["id"]
                if identifier=="example1":
                    example_1=chunk["description"]
                elif identifier=="example2":
                    example_2=chunk["description"]
                elif identifier=="example3":
                    example_3=chunk["description"]
                elif identifier=="format":
                    format=chunk["description"]
                else:
                    existing_unit = EvaluationUnit.objects.filter(file=eval_file, identifier=identifier).first()
                    if existing_unit:
                        existing_unit.delete()
                    original_description = chunk["description"]

                    EvaluationUnit.objects.create(
                        file=eval_file,
                        identifier=identifier,
                        description=original_description,
                        transformed_description="",
                        score=0.0,
                        scores=[],
                        linked_feature_ids=[],
                        linked_code=chunk["code_artifacts"]
                    )
                    add_to_vectordb(collection, original_description, "requirement", identifier, "requirement")
            
            req_id=0
            #########################################
            # Creating req from fetaures
            #########################################
            if use_feature:
                existing_file = EvaluationFile.objects.filter(project=project, name="Code_Features").first()
                if existing_file:
                    existing_units = EvaluationUnit.objects.filter(file=existing_file)
                    for existing_unit in existing_units:
                        existing_unit.delete()
                    existing_file.delete()
                code_eval = EvaluationFile.objects.create(project=project,name="Code_Features",path="")
                
                reqs, related_classes=get_reqs_from_feature(model, top_k, max_tokens, graph, EX, example_1, example_2, example_3, format, vector_collection_requirements, use_rag)
                
                
                for i, abstracted_req in enumerate(reqs):
                    req_id+=1
                    EvaluationUnit.objects.create(
                        file=code_eval,
                        identifier="feature_req_"+str(req_id),
                        description=abstracted_req.description,
                        transformed_description=abstracted_req.description,
                        score=0.0,
                        scores=[],
                        linked_feature_ids=[],
                        linked_code=[]
                    )
            #########################################
            # Creating req from Code (for evaluation)
            #########################################
            if use_code:
                existing_file = EvaluationFile.objects.filter(project=project, name="Req_from_code").first()
                if existing_file:
                    existing_units = EvaluationUnit.objects.filter(file=existing_file)
                    for existing_unit in existing_units:
                        existing_unit.delete()
                    existing_file.delete()
                code_eval = EvaluationFile.objects.create(project=project,name="Req_from_code",path="")
                reqs = all_in_context(model, top_k, REPO_ROOT_PATH, example_1, example_2, example_3, format, use_rag, vector_collection_requirements, MAX_TOKENS=max_tokens)
                for abstracted_req in reqs:
                    req_id+=1
                    EvaluationUnit.objects.create(
                        file=code_eval,
                        identifier="code_req_"+str(req_id),
                        description=abstracted_req,
                        transformed_description=abstracted_req,
                        score=0.0,
                        scores=[],
                        linked_feature_ids=[],
                        linked_code=[]
                    )

            if rag:
                use_rag=True
                #########################################
                # Creating req from fetaures RAG
                #########################################
                if use_feature:
                    existing_file = EvaluationFile.objects.filter(project=project, name="Code_Features_RAG").first()
                    if existing_file:
                        existing_units = EvaluationUnit.objects.filter(file=existing_file)
                        for existing_unit in existing_units:
                            existing_unit.delete()
                        existing_file.delete()
                    code_eval = EvaluationFile.objects.create(project=project,name="Code_Features_RAG",path="")
                    
                    reqs, related_classes=get_reqs_from_feature(model, top_k, max_tokens, graph, EX, example_1, example_2, example_3, format, vector_collection_requirements, use_rag)
                    
                    for i, abstracted_req in enumerate(reqs):
                        req_id+=1
                        EvaluationUnit.objects.create(
                            file=code_eval,
                            identifier="feature_RAG_req_"+str(req_id),
                            description=abstracted_req.description,
                            transformed_description=abstracted_req.description,
                            score=0.0,
                            scores=[],
                            linked_feature_ids=[],
                            # linked_code=related_classes[i]
                            linked_code=[]
                        )
                #########################################
                # Creating req from Code (for evaluation) RAG
                #########################################
                if use_code:
                    existing_file = EvaluationFile.objects.filter(project=project, name="Req_from_code_RAG").first()
                    if existing_file:
                        existing_units = EvaluationUnit.objects.filter(file=existing_file)
                        for existing_unit in existing_units:
                            existing_unit.delete()
                        existing_file.delete()
                    code_eval = EvaluationFile.objects.create(project=project,name="Req_from_code_RAG",path="")
                    reqs = all_in_context(model, top_k, REPO_ROOT_PATH, example_1, example_2, example_3, format, use_rag, vector_collection_requirements, MAX_TOKENS=max_tokens)
                    for abstracted_req in reqs:
                        req_id+=1
                        EvaluationUnit.objects.create(
                            file=code_eval,
                            identifier="code_RAG_req_"+str(req_id),
                            description=abstracted_req,
                            transformed_description=abstracted_req,
                            score=0.0,
                            scores=[],
                            linked_feature_ids=[],
                            linked_code=[]
                        )
