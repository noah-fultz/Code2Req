#########
# Views #
#########

def get_edges_sparql(where_clause):
    return f"""
    SELECT DISTINCT ?source ?target ?label WHERE {{
        {where_clause}
    }}
    """

def get_nodes_sparql(class_filter):
    return f"""
    SELECT DISTINCT ?id ?label ?type WHERE {{
        ?id a ?type ;
            :name ?label {class_filter}

          # Exclude superclasses if there's a more specific type
        FILTER NOT EXISTS {{
            ?id a ?type2 .
            ?type2 rdfs:subClassOf+ ?type .
            FILTER(?type2 != ?type)
        }}
    }}
    """

def get_node_description_sparql(uri):
    return f"""
        SELECT ?description ?code ?score ?level ?type WHERE {{
        OPTIONAL {{ <{uri}> :description ?description. }}
        OPTIONAL {{ <{uri}> :code ?code. }}
        OPTIONAL {{ <{uri}> :score ?score. }}
        OPTIONAL {{ <{uri}> :level ?level. }}
        OPTIONAL {{ <{uri}> a ?type. }}
        FILTER NOT EXISTS {{
            ?id a ?type2 .
            ?type2 rdfs:subClassOf+ ?type .
            FILTER(?type2 != ?type)
        }} 
    }}"""

def get_network_for_uri_sparql(uri):
    uris_clause = " ".join(f"<{u}>" for u in uri)
    return f"""
    SELECT DISTINCT ?sourceId ?sourceName ?sourceType ?targetId ?targetName ?targetType ?relation
    WHERE {{
    VALUES ?input {{ {uris_clause} }}

    {{
        # Edges involving the input nodes (in both directions)
        {{
        BIND(?input AS ?sourceId)
        ?sourceId ?relation ?targetId .
        }} UNION {{
        ?sourceId ?relation ?input .
        BIND(?input AS ?targetId)
        }}

        ?sourceId :name ?sourceName .
        ?sourceId a ?sourceType .
        ?targetId :name ?targetName .
        ?targetId a ?targetType .
    }}

    UNION

    {{
        # Get neighbors of each input node
        {{
        ?input ?r1 ?n1 .
        }} UNION {{
        ?n1 ?r1 ?input .
        }}

        {{
        ?input ?r2 ?n2 .
        }} UNION {{
        ?n2 ?r2 ?input .
        }}

        # Link between neighbors
        ?n1 ?relation ?n2 .
        FILTER(?n1 != ?n2)
        FILTER(?n1 != ?input && ?n2 != ?input)

        BIND(?n1 AS ?sourceId)
        BIND(?n2 AS ?targetId)

        ?sourceId :name ?sourceName .
        ?sourceId a ?sourceType .
        ?targetId :name ?targetName .
        ?targetId a ?targetType .
    }}

    FILTER NOT EXISTS {{
        ?id a ?type2 .
        ?type2 rdfs:subClassOf+ ?targetType .
        FILTER(?type2 != ?targetType)
    }}
    FILTER NOT EXISTS {{
        ?id a ?type3 .
        ?type3 rdfs:subClassOf+ ?sourceType .
        FILTER(?type3 != ?sourceType)
    }}
    }}
    """
   