import ollama
import uuid
import chromadb
from django.shortcuts import get_object_or_404
from c2c_web.models import Project
from SPARQLWrapper import SPARQLWrapper, JSON
from django.conf import settings
import numpy as np
from rdflib import URIRef
from types import SimpleNamespace


def embed(str):
    print(len(str))
    print(str)
    response = ollama.embed(model="mxbai-embed-large", input=str)
    embeddings = response["embeddings"]
    return embeddings

def add_to_vectordb(collection, doc, source, source_class, type_, score=""):
    chunk_size = 1000
    overlap = 150
    i = 0
    n = len(doc)
    
    while i < n:
        end = min(i + chunk_size, n)
        chunk = doc[i:end]
        emb = embed(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=emb,
            metadatas=[{
                "source": source,
                "source_class": source_class,
                "type": type_,
                "score": score,
                "char_start": i,
                "char_end": end
            }],
            documents=[chunk]
        )
        if end == n:
            break
        i += chunk_size - overlap


def get_from_vectordb(collection_name, source_id, include=["documents"]):
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_collection(name=collection_name)
    result = collection.get(
        where={"source": source_id},
        include=include
    )
    res={}
    for inc in include:
        res[inc]=result[inc][0]
    return res

def get_all_from_vectordb(collection_name, include=["documents", "metadatas"]):
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_collection(name=collection_name)
    results = collection.get(include=include)
    return results

def query_vector_db(embedded_query, vector_collection, top_n=5, include=["distances", "metadatas", "documents"]):
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_collection(name=vector_collection)
    results = collection.query(
        query_embeddings=embedded_query,
        n_results=top_n,
        include=include
    )
    return results

def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
             

def get_or_create_collection(client, collection_name):
    try:
        # Try to get the existing collection
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception as e:
        # If not found, create a new one
        print(f"Collection '{collection_name}' not found. Creating new.")
        collection = client.create_collection(name=collection_name)
    return collection

def delete_collection(client, collection_name):
    try:
        client.delete_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection '{collection_name}' does not exist")

def get_project_by_id(project_id):
    if project_id:
        project = get_object_or_404(Project, id=project_id)
    else:
        # project = None
        project = SimpleNamespace(
            id=0,
            name="",
            github_repo="",
            requirement_file=None,
            repository_analyzed=False,
            requirements_analyzed=False,
            lines_of_code=0,
            functions=0,
            classes=0,
            snippets=0,
            imports=0,
            artifacts=0,
            codefiles=0,
            features=0,
            created_at=None,
            updated_at=None,
            is_fake=True,  # useful downstream flag
            clusters = 10,
            top_k = 5,
            max_tokens = 400000,
            model = "gpt-5",
            use_rag = True,
            use_feature = True,
            use_code =True,
        )
    return project


def get_project_by_session(request):
    project_id = request.session.get('selected_project_id', 1)
    project = Project.objects.filter(id=project_id).first()

    if not project:
        # Create a fake "empty" project-like object
        project = SimpleNamespace(
            id=0,
            name="NoProjectSelected",
            github_repo="",
            requirement_file=None,
            repository_analyzed=False,
            requirements_analyzed=False,
            lines_of_code=0,
            functions=0,
            classes=0,
            snippets=0,
            imports=0,
            artifacts=0,
            codefiles=0,
            features=0,
            created_at=None,
            updated_at=None,
            is_fake=True,  # useful downstream flag
            clusters = 10,
            top_k = 5,
            max_tokens = 400000,
            model = "gpt-5",
            use_rag = True,
            use_feature = True,
            use_code =True,
        )

    return project

def get_all_projects():
    projects = Project.objects.all().order_by('name')
    return projects


def get_vector_collection_names(project_name):
    vector_collection_requirements="requirementsVectors_"+project_name
    vector_collection_feature="featureVectors_"+project_name
    vector_collection_code="codeVectors_"+project_name
    return vector_collection_code, vector_collection_feature, vector_collection_requirements

def query_graphdb(query, project_name):
    sparql = SPARQLWrapper(settings.SPARQL_ENDPOINT+project_name)
    sparql.setQuery(settings.SPARQL_PREFIXES + query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

def get_exists(node_uri, graph):
    uri = URIRef(node_uri)
    exists = any(graph.triples((uri, None, None)))
    return exists

def get_code_artifacts_for_file(file_uri, graph, EX):
    file_ref = URIRef(file_uri)
    return list(graph.objects(file_ref, EX.containsArtifact))





   