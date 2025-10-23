import chromadb
from rdflib import Namespace, RDF
from c2c_web.utils.utils import add_to_vectordb, get_or_create_collection, delete_collection
import tiktoken
from c2c_web.utils.process.process_req.process_eval_file import read_code, get_all_features, get_tokens
from pathlib import Path

def get_stats(graph, EX, project, lines_of_code, repo_root):
    functions = len(list(graph.subjects(RDF.type, EX.Function)))
    classes = len(list(graph.subjects(RDF.type, EX.Class)))
    snippets = len(list(graph.subjects(RDF.type, EX.Snippet)))
    imports = len(list(graph.subjects(RDF.type, EX.Imports)))
    artifacts=functions+classes+snippets+imports
    codefiles = len(list(graph.subjects(RDF.type, EX.CodeFile)))
    features = len(list(graph.subjects(RDF.type, EX.Feature)))
    
    REPO_ROOT_PATH = Path(repo_root).resolve()
    code_list_per_file = {}
    code_list = read_code(REPO_ROOT_PATH, code_list_per_file)
    code_tokens=0
    for c in code_list:
        code_tokens+=get_tokens(code_list[c])
    
    all_features=get_all_features(graph, EX)
    feature_tokens = get_tokens(str(all_features))

    project.lines_of_code = lines_of_code
    project.functions = functions
    project.classes = classes
    project.snippets = snippets
    project.imports = imports
    project.artifacts = artifacts
    project.codefiles = codefiles
    project.features = features
    project.code_tokens = code_tokens
    project.feature_tokens = feature_tokens

def embed_features_desc_code(graph, ex, vector_path, vector_collection_code, vector_collection_feature):
    EX = Namespace(ex+"#")
    client = chromadb.PersistentClient(path=vector_path)
    delete_collection(client, vector_collection_code)
    collection = get_or_create_collection(client, vector_collection_code)

    all_artifacts = (
        list(graph.subjects(RDF.type, EX.Function)) +
        list(graph.subjects(RDF.type, EX.Class)) +
        list(graph.subjects(RDF.type, EX.Snippet)) +
        list(graph.subjects(RDF.type, EX.Imports))
    )

    for i, artifact in enumerate(all_artifacts):
        desc = graph.value(subject=artifact, predicate=EX.description)
        code = graph.value(subject=artifact, predicate=EX.code)
        if desc is not None:
            add_to_vectordb(collection, desc, artifact, "CodeArtifact", "description")
        if code is not None:
            add_to_vectordb(collection, code, artifact, "CodeArtifact", "code")

    delete_collection(client, vector_collection_feature)
    collection = get_or_create_collection(client, vector_collection_feature)
    all_features = list(graph.subjects(RDF.type, EX.Feature)) 
    for i, feature in enumerate(all_features):
        desc = graph.value(subject=feature, predicate=EX.description)
        if desc is not None:
            add_to_vectordb(collection, desc, feature, "Feature", "description")
