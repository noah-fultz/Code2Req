
from pathlib import Path
from rdflib import RDF, Graph, Literal, URIRef, Namespace
from c2c_web.utils.prompts import Feature, get_tag_features_sys_prompt, get_tag_features_user_prompt
from c2c_web.utils.utils import cosine_similarity, embed
from c2c_web.utils.call_ollama import safe_ollama_chat
from typing import List, Tuple
from infomap import Infomap
from collections import defaultdict
from ollama import chat
import hashlib
import re
import tiktoken

from rdflib import Graph, Namespace, RDF
from typing import List, Tuple, Set, Optional


from rdflib import URIRef

def get_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def are_in_same_codefile(member1: str, member2: str, graph, EX) -> bool:
    uri1 = URIRef(member1)
    uri2 = URIRef(member2)
    # Find all CodeFiles that contain each artifact
    files1 = {
        str(codefile)
        for codefile in graph.subjects(predicate=EX.containsArtifact, object=uri1)
    }
    files2 = {
        str(codefile)
        for codefile in graph.subjects(predicate=EX.containsArtifact, object=uri2)
    }
    # Check for intersection
    in_same_file = not files1.isdisjoint(files2)
    return in_same_file

def get_embedded(graph: Graph,EX: Namespace):
    embeddings = {}
    code_artifact_classes = {EX.Function, EX.Class, EX.Snippet}

    # Create a fast lookup set for all allowed artifacts
    valid_artifacts = {
        str(s)
        for class_type in code_artifact_classes
        for s in graph.subjects(RDF.type, class_type)
    }

    for s, p, o in graph.triples((None, EX.usesArtifact, None)):
        s_str, o_str = str(s), str(o)
        if s_str in valid_artifacts and o_str in valid_artifacts:
            if s_str not in embeddings:
                code = graph.value(subject=URIRef(s_str), predicate=EX.code)
                embeddings[s_str]=1#embed(code)[0]
            if o_str not in embeddings:
                code = graph.value(subject=URIRef(o_str), predicate=EX.code)
                embeddings[o_str]=1#embed(code)[0]

    return embeddings

def get_artifact_edgelist(
    graph: Graph,
    EX: Namespace,
    all_embeddings,
    limit_by_uris: Optional[List[str]] = None
) -> Tuple[List[Tuple[str, str]], int]:
    edgelist = []
    code_artifact_classes = {EX.Function, EX.Class, EX.Snippet}

    # Create a fast lookup set for all allowed artifacts
    valid_artifacts = {
        str(s)
        for class_type in code_artifact_classes
        for s in graph.subjects(RDF.type, class_type)
    }

    # Optional filtering set
    limit_set = set(limit_by_uris) if limit_by_uris else None

    for s, p, o in graph.triples((None, EX.usesArtifact, None)):
        s_str, o_str = str(s), str(o)
        if s_str in valid_artifacts and o_str in valid_artifacts:
            if limit_set and (s_str not in limit_set or o_str not in limit_set):
                continue
            # similarity = cosine_similarity(all_embeddings[s_str], all_embeddings[o_str])
            in_same_file = are_in_same_codefile(s_str, o_str, graph, EX) 
            if in_same_file:
                edgelist.append((s_str, o_str, 1))
            else:
                edgelist.append((s_str, o_str, 0.1))

    return edgelist


def get_artifact_cluster_descriptions(cluster_members, graph, EX):
    all_desc = {}
    for member in cluster_members:
        member_uri=URIRef(member)
        desc = graph.value(subject=member_uri, predicate=EX.description)
        code = graph.value(subject=member_uri, predicate=EX.code)
        if code:
            all_desc[str(member)] = str(code)
        else:
            print("WARN!!! "+member+" has no code")
    return all_desc


def get_description(all_desc, type, model):
    messages=[{'role': 'system', 'content': get_tag_features_sys_prompt(type)}]
    messages.append({'role': 'user', 'content': get_tag_features_user_prompt(str(all_desc))})
    print("Sending tokens: "+str(get_tokens(get_tag_features_user_prompt(str(all_desc)))))
    response = safe_ollama_chat(model=model,messages=messages,format=Feature)
    feature = Feature.model_validate_json(response.message.content)
    similarity = cosine_similarity(embed(" ".join(str(v) for v in all_desc.values()))[0], embed(feature.description)[0])
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

def run_infomap(edge_list, preferred_cluster_size, force_split=False):
    uri_to_id = {uri: idx for idx, uri in enumerate(set([n for edge in edge_list for n in edge]))}
    id_to_uri = {v: k for k, v in uri_to_id.items()}

    if force_split:
        desired_modules = int(len(id_to_uri) / (preferred_cluster_size / 4))
        infomap = Infomap(f"--directed --seed 42 --silent --preferred-number-of-modules {desired_modules}")
    else:
        infomap = Infomap("--directed --seed 42 --silent")

    for src, tgt, score in edge_list:
        infomap.addLink(uri_to_id[src], uri_to_id[tgt], score)

    infomap.run()
    clusters = defaultdict(list)
    for node in infomap.nodes:
        cluster_id = node.module_id
        uri = id_to_uri[node.node_id]
        clusters[cluster_id].append(uri)

    return clusters

def recursive_cluster_maker(
    final_cluster_map, edge_list, all_embeddings, preferred_cluster_size,
    graph, EX, max_depth, depth, prev_clusters=None
):
    force_split = False
    current_clusters = run_infomap(edge_list, preferred_cluster_size, force_split)

    # Check if clustering is the same as previous
    if prev_clusters is not None:
        same = True
        if len(current_clusters) != len(prev_clusters):
            same = False
        else:
            for cid in current_clusters:
                if cid not in prev_clusters or set(current_clusters[cid]) != set(prev_clusters[cid]):
                    same = False
                    break
        if same:
            force_split = True
            current_clusters = run_infomap(edge_list, preferred_cluster_size, force_split)

    depth += 1
    singles = []
    valid_cluster_ids = []

    for cid in current_clusters:
        members = current_clusters[cid]

        if len(members) > preferred_cluster_size:
            edge_list_cluster = get_artifact_edgelist(graph, EX, all_embeddings, members)

            if not edge_list_cluster:
                cluster_id = len(final_cluster_map) + 1
                final_cluster_map[cluster_id] = {
                    "depth": depth,
                    "members": members,
                    "children": []
                }
                valid_cluster_ids.append(cluster_id)
            else:
                child_ids = recursive_cluster_maker(
                    final_cluster_map,
                    edge_list_cluster,
                    all_embeddings,
                    preferred_cluster_size,
                    graph,
                    EX,
                    max_depth,
                    depth,
                    prev_clusters={cid: members}
                )
                if len(child_ids) > 1:
                    cluster_id = len(final_cluster_map) + 1
                    final_cluster_map[cluster_id] = {
                        "depth": depth,
                        "members": [],
                        "children": child_ids
                    }
                    valid_cluster_ids.append(cluster_id)
                else:
                    valid_cluster_ids.extend(child_ids)

        elif len(members) == 1:
            singles.append(members[0])

        else:
            cluster_id = len(final_cluster_map) + 1
            final_cluster_map[cluster_id] = {
                "depth": depth,
                "members": members,
                "children": []
            }
            valid_cluster_ids.append(cluster_id)

    if singles:
        cluster_id = len(final_cluster_map) + 1
        final_cluster_map[cluster_id] = {
            "depth": depth,
            "members": singles,
            "children": []
        }
        valid_cluster_ids.append(cluster_id)

    return valid_cluster_ids


def cluster_code_artifacts(graph: Graph, ex: str, repo_root: str, model:str, preferred_cluster_size=10, hierarchical=False):
    EX = Namespace(ex + "#")
    REPO_ROOT_PATH = Path(repo_root).resolve()
    BASE_URI = f"http://example.org/{REPO_ROOT_PATH.name}"
    
    all_embeddings = get_embedded(graph, EX)
    edge_list = get_artifact_edgelist(graph, EX, all_embeddings)
    final_cluster_map = {}

    max_depth = 200
    depth = 0

    top_level_clusters = recursive_cluster_maker(
        final_cluster_map,
        edge_list,
        all_embeddings,
        preferred_cluster_size,
        graph,
        EX,
        max_depth,
        depth,
        prev_clusters=None  # initial call
    )

    # Optional: Create a root node to hold the top-level clusters
    final_cluster_map[len(final_cluster_map)+1] = {
        "depth": 0,
        "members": [],
        "children": top_level_clusters
    }

    print(len(final_cluster_map))
    total_nodes = 0
    for c in final_cluster_map:
        total_nodes += len(final_cluster_map[c]["members"])
    print("Total nodes:", total_nodes)
    print(len(final_cluster_map))

    feature_desc_map={}
    for c in final_cluster_map:
        print(f"ID: {c} Depth: {final_cluster_map[c]['depth']} num_members: {len(final_cluster_map[c]['members'])} num_children: {len(final_cluster_map[c]['children'])}")
        if final_cluster_map[c]["members"]!=[]:
            all_desc = get_artifact_cluster_descriptions(final_cluster_map[c]["members"], graph, EX)
            feature, similarity = get_description(all_desc, "Code", model)
            if hierarchical==False:
                final_cluster_map[c]['depth']=2
            feature_id = upload_features_to_graph(graph, feature, similarity, final_cluster_map[c]["members"], final_cluster_map[c]['depth'], EX, BASE_URI)
            feature_desc_map[str(c)]={"desc":feature.name+": "+feature.description, "uri":feature_id}
        elif hierarchical==True:
            all_desc={}
            child_uris=[]
            for f in final_cluster_map[c]["children"]:
                if str(f) in feature_desc_map:
                    all_desc[str(f)]=feature_desc_map[str(f)]["desc"]
                    child_uris.append(feature_desc_map[str(f)]["uri"])
            feature, similarity = get_description(all_desc, "Features", model)
            feature_id = upload_features_to_graph(graph, feature, similarity, child_uris, final_cluster_map[c]['depth'], EX, BASE_URI)
            feature_desc_map[str(c)]={"desc":feature.name+": "+feature.description, "uri":feature_id}
    print("Done with initial clusters")



