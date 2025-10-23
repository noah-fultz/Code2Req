from django.conf import settings
from django.shortcuts import get_object_or_404
from c2c_web.models import Project
from c2c_web.utils.process.process_repo.parse_repo import build_code_repository_graph
from c2c_web.utils.process.process_repo.get_stats_and_embed import embed_features_desc_code, get_stats
from c2c_web.utils.process.process_req.process_eval_file import process_eval_files
from c2c_web.utils.process.process_req.compare import compare
from c2c_web.utils.process.process_repo.cluster_code import cluster_code_artifacts
from c2c_web.utils.process.process_repo.join_features import join_features
from c2c_web.utils.utils import get_vector_collection_names
from rdflib import Graph, Namespace
import requests

def upload_ttl_to_graphdb(repo_url, ttl_path, graph_uri=None):
    with open(ttl_path, 'rb') as f:
        headers = {
            "Content-Type": "text/turtle"
        }
        params = {}
        if graph_uri:
            params["graph"] = graph_uri
        response = requests.post(
            f"{repo_url}/statements",
            data=f,
            headers=headers,
            params=params
        )
    print("Status code:", response.status_code)
    if response.status_code != 204:
        print("Error:", response.text)
    else:
        print("Upload successful")


def start_repository_analysis(project):
    project_name=project.name
    clusters = project.clusters
    model=project.model
    repo_dir="data/"+project_name+"/repo/"
    graph_file="data/"+project_name+"/ttl-files/output.ttl"
    graph_file_code_repo="data/"+project_name+"/ttl-files/output_code_repo_graph.ttl"
    graph_file_flat="data/"+project_name+"/ttl-files/output_infomap_flat.ttl"
    graph_db_url=settings.SPARQL_ENDPOINT+project_name
    vector_collection_code, vector_collection_feature,_ = get_vector_collection_names(project_name)
    
    # # 1. Extract and describe ALL code artifacts
    graph, lines_of_code = build_code_repository_graph(repo_dir, settings.GRAPH_URI)
    graph.serialize(graph_file_code_repo, format="turtle") # Storing in case something breaks 
    
    # # 2. Find call-graphs and make features
    graph = Graph()
    graph.parse(graph_file_code_repo, format="turtle")
    cluster_code_artifacts(graph, settings.GRAPH_URI, repo_dir, model, clusters)
    graph.serialize(graph_file_flat, format="turtle") # Storing in case something breaks 
    
    graph = Graph()
    graph.parse(graph_file_flat, format="turtle") 
    join_features(graph, settings.GRAPH_URI, repo_dir, model)
    graph.serialize(graph_file, format="turtle") # Storing in case something breaks 

    # 3. Upload model and graph to graphDB
    upload_ttl_to_graphdb(repo_url=graph_db_url, ttl_path="data/repoModel.ttl",graph_uri=settings.GRAPH_URI)
    upload_ttl_to_graphdb(repo_url=graph_db_url, ttl_path=graph_file,graph_uri=settings.GRAPH_URI)
    
    # # 4. Embed all text for artifacts and features and save project stats
    graph = Graph()
    graph.parse(graph_file, format="turtle")
    embed_features_desc_code(graph, settings.GRAPH_URI, settings.VECTOR_DB_PATH, vector_collection_code, vector_collection_feature)
    get_stats(graph, Namespace(settings.GRAPH_URI+"#"), project, lines_of_code, repo_dir)

def start_req_analysis(project):
    project_name=project.name
    model=project.model
    max_tokens=project.max_tokens
    top_k=project.top_k
    use_rag=project.use_rag
    use_feature=project.use_feature
    use_code=project.use_code

    graph_file="data/"+project_name+"/ttl-files/output.ttl"
    repo_dir="data/"+project_name+"/repo/"
    _, _,vector_collection_requirements = get_vector_collection_names(project_name)
    comparison_report_path="data/"+project_name+"/comparisons/"
    
    graph = Graph()
    graph.parse(graph_file, format="turtle")
    # Process evaluation docs
    process_eval_files(project, graph, settings.GRAPH_URI, repo_dir, vector_collection_requirements, settings.VECTOR_DB_PATH, model, top_k, max_tokens, use_rag, use_feature, use_code)
    # compare
    compare(project, comparison_report_path)

def analyze(project_id, analysis_type):
    project = get_object_or_404(Project, id=project_id)
    if analysis_type == 'repository':
        project.repository_analyzed = True
        start_repository_analysis(project)
    elif analysis_type == 'evaluation':
        project.requirements_analyzed = True
        start_req_analysis(project)
    else:
        return "ERROR"
    project.save()