from c2c_web.utils.utils import query_graphdb, embed, query_vector_db, get_vector_collection_names
from c2c_web.utils.sparql_queries import get_edges_sparql, get_nodes_sparql, get_network_for_uri_sparql

PREDEFINED_CLASS_INDEX = {
    "RepositoryArtifact": "type:0",
    "CodeRepository": "type:1",
    "CodeDirectory": "type:2",
    "CodeFile": "type:3",
    "CodeArtifact": "type:4",
    "Function": "type:5",
    "Class": "type:6",
    "Snippet": "type:7",
    "Imports": "type:8",
    "Feature": "type:9",
}

def get_from_vector_db(input, n, project_name):
    vector_collection_code, vector_collection_feature,_ = get_vector_collection_names(project_name)
    embedded_query = embed(input)
    results_code = query_vector_db(embedded_query, vector_collection_code, n)
    results_feature = query_vector_db(embedded_query, vector_collection_feature, n)
    dist_code=results_code["distances"][0][0]
    dist_feature=results_feature["distances"][0][0]
    if dist_code<dist_feature:
        return results_code["metadatas"][0][0]["source"]
    else:
        return results_feature["metadatas"][0][0]["source"]

def get_network_for_id(node_name, project):
    query= get_network_for_uri_sparql(node_name)
    print(query)
    results = query_graphdb(query, project.name)
    nodes = []
    edges = []

    for row in results:
        type_name_a = row["sourceType"]["value"].split("#")[-1]
        nodes.append({
            "data": {
                "id": row["sourceId"]["value"],
                "label": row["sourceName"]["value"],
                "type": type_name_a,
                "styleClass": PREDEFINED_CLASS_INDEX[type_name_a]   # used in Cytoscape
            }
        })
        type_name_b = row["targetType"]["value"].split("#")[-1]
        nodes.append({
            "data": {
                "id": row["targetId"]["value"],
                "label": row["targetName"]["value"],
                "type": type_name_b,
                "styleClass": PREDEFINED_CLASS_INDEX[type_name_b]  # used in Cytoscape
            }
        })
        edge_id = row["sourceId"]["value"]+"___"+row["relation"]["value"].split("#")[-1]+"___"+row["targetId"]["value"]
        edges.append({
            "data": {
                "id": edge_id,
                "label": row["relation"]["value"].split("#")[-1],
                "source": row["sourceId"]["value"],
                "target": row["targetId"]["value"]
            }
        })
    return nodes, edges

def get_nodes(node_type, project_name):
    if node_type == "code":
        class_filter = ". ?type rdfs:subClassOf* :CodeArtifact ."
    elif node_type == "features":
        class_filter = ". FILTER(?type = :Feature)"
    elif node_type == "codefeatures":
        class_filter = ". { ?type rdfs:subClassOf* :CodeArtifact .} UNION { FILTER(?type = :Feature) }    {?id (:usesArtifact|^:usesArtifact|:isFeatureFor|^:isFeatureFor) ?any .}"
    else:  # all
        class_filter = ". "

    query = get_nodes_sparql(class_filter)
    results = query_graphdb(query, project_name)
    nodes = []
    for r in results:
        node_id = r["id"]["value"]
        label = r["label"]["value"]
        full_type = r["type"]["value"]
        type_name = full_type.split("#")[-1]

        nodes.append({
            "data": {
                "id": node_id,
                "label": label,
                "type": type_name,
                "styleClass": PREDEFINED_CLASS_INDEX[type_name]  # used in Cytoscape
            }
        })
    return nodes

def get_edges(project_name, node_type="all"):
    if node_type == "code":
        predicate_filter = ":usesArtifact"
    elif node_type == "features":
        predicate_filter = ":isFeatureFor"
    else:  # all
        predicate_filter = None
    # Dynamically build the WHERE clause
    if predicate_filter:
        where_clause = f"?source {predicate_filter} ?target . BIND({predicate_filter} AS ?label)"
    else:
        where_clause = """
            ?source ?label ?target .
        """
    query = get_edges_sparql(where_clause)
    results = query_graphdb(query, project_name)
    return [
        {
            "data": {
                "id": r["source"]["value"]+"___"+r["label"]["value"].split("#")[-1]+"___"+r["target"]["value"],
                "source": r["source"]["value"],
                "target": r["target"]["value"],
                "label": r["label"]["value"].split("#")[-1]  # gets "usesArtifact" or "isFeatureFor"
            }
        }
        for r in results
    ]

def split_uris(query):
    return query.split(",")

def expand_network(request, project):
    query_param = request.GET.get("query")
    q = split_uris(query_param.replace("QUERY:",""))
    nodes, edges = get_network_for_id(q, project)
    return nodes, edges


def get_network(request, project):
    query_param = request.GET.get("query")
    print("QUERY:")
    print(query_param)
    nodes=[]
    edges=[]

    if project.name=="NoProjectSelected":
        return nodes, edges

    if query_param==None: # Default for page init
        query_param="FILTER:features"

    if query_param.startswith("QUERY:http://example.org/repo") or query_param.startswith("http://example.org/repo"):
        print(query_param)
        q = split_uris(query_param.replace("QUERY:",""))
        nodes, edges = get_network_for_id(q, project)
    elif query_param.startswith("QUERY:"):
        print(query_param)
        print("MAING VECTOR SEARCH")
        q = query_param.replace("QUERY:","")
        node_name = get_from_vector_db(q, 1, project.name)
        nodes, edges = get_network_for_id([node_name], project)
    elif query_param.startswith("FILTER:"):
        q=query_param.replace("FILTER:","").replace("/","").strip()
        nodes = get_nodes(q, project.name)
        node_ids = {node["data"]["id"] for node in nodes}
        all_edges = get_edges(project.name, q)
        edges = [
            edge for edge in all_edges
            if edge["data"]["source"] in node_ids and edge["data"]["target"] in node_ids
        ]

    return nodes, edges
