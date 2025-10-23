from pathlib import Path
import hashlib
import re
from rdflib import RDF, Graph, Literal, URIRef, Namespace
from ollama import chat
from c2c_web.utils.prompts import Feature, get_tag_features_sys_prompt, get_tag_features_user_prompt
from c2c_web.utils.utils import cosine_similarity as cos, embed
from c2c_web.utils.call_ollama import safe_ollama_chat
from rdflib import RDF

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from infomap import Infomap


import networkx as nx
from rdflib import URIRef

def build_codeartifact_graph(rdf_graph, EX):
    G = nx.Graph()
    for s, p, o in rdf_graph.triples((None, EX.usesArtifact, None)):
        G.add_edge(str(s), str(o))

    for feature, _, artifact in rdf_graph.triples((None, EX.isFeatureFor, None)):
        G.add_edge(str(feature), str(artifact))
    return G

def feature_shortest_paths(rdf_graph, feature_descriptions, EX):
    G = build_codeartifact_graph(rdf_graph, EX)
    artifact_paths = dict(nx.all_pairs_shortest_path_length(G))
    path_matrix = {}
    for i, f1 in enumerate(feature_descriptions):
        path_matrix[f1] = {}
        for j, f2 in enumerate(feature_descriptions):
            if i == j:
                path_matrix[f1][f2] = 0
            else:
                dist = artifact_paths.get(f1, {}).get(f2, float('inf'))
                path_matrix[f1][f2] = dist 

    return path_matrix

def save_embeddings(embeddings, filename="feature_embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(filename="feature_embeddings.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
         
def get_all_features(graph, EX) -> dict:
    """Extracts all Features and their name + description from an RDFLib graph."""
    feature_map = {}
    for feature in graph.subjects(RDF.type, EX.Feature):
        name = graph.value(feature, EX.name)
        desc = graph.value(feature, EX.description)
        if name and desc:
            feature_map[str(feature)] = f"{str(name)}: {str(desc)}"
    return feature_map

def extract_name(value):
    match = re.search(r'#(.*?)(?:_[a-f0-9]{10,}|)_feature$', value)
    if match:
        ex = match.group(1)
        return ex.replace("_"," ")
    
def get_feature_embeddings(feature_dict):
    feature_embeddings={}
    for f in feature_dict:
        embedding = embed(extract_name(f)+": "+feature_dict[f])[0]
        feature_embeddings[f]=embedding
    return feature_embeddings

def compute_similarity_matrix(all_embeddings: dict) -> pd.DataFrame:
    """Returns a cosine similarity matrix as a Pandas DataFrame."""
    feature_ids = list(all_embeddings.keys())
    embedding_matrix = np.array([all_embeddings[fid] for fid in feature_ids])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(embedding_matrix)

    # Create a labeled DataFrame
    return pd.DataFrame(sim_matrix, index=feature_ids, columns=feature_ids)

def similarity_to_infomap_edges(
    sim_df: pd.DataFrame,
    path_matrix: dict,
    threshold: float = 0.6,
    alpha: float = 1.0
):
   
    feature_ids = sim_df.index.tolist()
    n = len(feature_ids)
    raw_edges = []

    # Step 1: collect all finite path lengths for rescaling
    finite_paths = []
    for f1 in feature_ids:
        for f2 in feature_ids:
            if f1 == f2:
                continue
            d = path_matrix.get(f1, {}).get(f2, float('inf'))
            if np.isfinite(d):
                finite_paths.append(d)

    if not finite_paths:
        return [], feature_ids  # no connections

    min_path, max_path = min(finite_paths), max(finite_paths)
    rescale_path = lambda d: (d - min_path) / (max_path - min_path) if max_path > min_path else 0.0
    # Step 2: compute adjusted similarity using rescaled path
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = feature_ids[i], feature_ids[j]
            sim = sim_df.iat[i, j]

            path_len = path_matrix.get(f1, {}).get(f2, float('inf'))
            if not np.isfinite(path_len):
                continue

            rescaled_len = rescale_path(path_len)
            adjusted_sim = sim * np.exp(-alpha * rescaled_len)
            # print("Path length: "+str(path_len)+" -> "+str(rescaled_len))
            # print("Semantic similarity: "+str(sim))
            # print("SCORE: "+str(adjusted_sim))
            # print("")
            if adjusted_sim>=threshold:
                raw_edges.append((i, j, adjusted_sim))

    return raw_edges, feature_ids



def run_infomap_on_similarity(edges, feature_ids, num_clusters):
    """Run Infomap on a weighted edge list and return cluster mapping."""
    infomap = Infomap(silent=True, directed=False)

    for src, tgt, weight in edges:
        infomap.addLink(src, tgt, weight)
    infomap.run()
    clusters = {}
    for node in infomap.nodes:
        feature_uri = feature_ids[node.node_id]
        module_id = node.module_id
        clusters.setdefault(module_id, []).append(feature_uri)

    return clusters

def get_description(all_desc, type, model):
    messages=[{'role': 'system', 'content': get_tag_features_sys_prompt(type)}]
    messages.append({'role': 'user', 'content': get_tag_features_user_prompt(str(all_desc))})
    response = safe_ollama_chat(model=model,messages=messages,format=Feature)
    feature = Feature.model_validate_json(response.message.content)
    similarity = cos(embed(" ".join(str(v) for v in all_desc.values()))[0], embed(feature.description)[0])
    print(f"Semantic similarity: {similarity:.4f}")
    return feature, similarity

def make_stackfeature_id(feature, members):
    key = feature.name + "::" + ",".join(sorted(members))
    hash_id = hashlib.sha256(key.encode()).hexdigest()[:10]
    clean_name = re.sub(r'[^A-Za-z0-9_]', '', feature.name.replace(' ', '_'))
    return f"{clean_name}_{hash_id}_feature"

def upload_features_to_graph(graph: Graph, feature, score, members, level, EX:Namespace, BASE_URI):
    feature_id = make_stackfeature_id(feature, members)
    feature_uri = URIRef(f"{BASE_URI}#{feature_id}")
    graph.add((feature_uri, EX.name, Literal(feature.name)))
    graph.add((feature_uri, EX.description, Literal(feature.description)))
    graph.add((feature_uri, EX.score, Literal(f"Cosine similarity on embeddings: {score:.4f}")))
    graph.add((feature_uri, EX.level, Literal(str(level))))
    graph.add((feature_uri, RDF.type, EX.Feature))
    for artifact_uri in members:
        graph.add((feature_uri, EX.isFeatureFor, URIRef(artifact_uri)))
    return str(feature_uri)


def join_features(graph, ex, repo_root, model):
    EX = Namespace(ex+"#")
    REPO_ROOT_PATH = Path(repo_root).resolve()
    BASE_URI = f"http://example.org/{REPO_ROOT_PATH.name}"

    feature_descriptions = get_all_features(graph, EX)
    feature_embeddings = get_feature_embeddings(feature_descriptions)
    num_clusters=len(feature_descriptions)/5
    shortest_paths=feature_shortest_paths(graph, feature_descriptions, EX)
    similarity_df = compute_similarity_matrix(feature_embeddings)
    edges, feature_ids = similarity_to_infomap_edges(similarity_df, shortest_paths)
    clusters = run_infomap_on_similarity(edges, feature_ids, num_clusters)
    
    total_nodes=0
    descs_for_top_level={}
    used_features=[]
    for cid, members in clusters.items():
        print(f"Cluster {cid}: {len(members)}")
        all_desc={}
        for f in feature_descriptions:
            if f in members:
                used_features.append(f)
                all_desc[f]=feature_descriptions[f]
        total_nodes+=len(members)

        feature, similarity = get_description(all_desc, "Features", model)
        feature_id = upload_features_to_graph(graph, feature, similarity, members, 1, EX, BASE_URI)
        descs_for_top_level[feature_id]=feature.name+": "+feature.description
    
    for f in feature_descriptions:
        if f not in used_features:
            descs_for_top_level[f]=feature_descriptions[f]
    feature, similarity = get_description(descs_for_top_level, "Features", model)
    feature_id = upload_features_to_graph(graph, feature, similarity, list(descs_for_top_level.keys()), 0, EX, BASE_URI)
    print("Total features: "+str(total_nodes))
    
