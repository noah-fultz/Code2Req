from urllib.parse import urlparse
import re
import csv
from c2c_web.models import EvaluationUnit
from c2c_web.utils.utils import get_vector_collection_names, query_graphdb, get_from_vectordb
from c2c_web.utils.sparql_queries import get_node_description_sparql
    
def get_regulation_text(regulation_id, filepath):
    content = [{"title": "ID", "content": regulation_id}]
    print(regulation_id)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(regulation_id + ":"):
                    # Split the line into ID and text
                    _, text = line.split(":", 1)
                    content.append({"title": "Regulatory Guideline", "content": text.strip()})
                    return content

        # If no matching line found
        content.append({"title": "Error", "content": "Regulation not found."})
        return content

    except Exception as e:
        content.append({"title": "Error", "content": f"Error parsing regulation: {str(e)}"})
        return content

def get_artifact_details(project, id):
    queryset = EvaluationUnit.objects.filter(
        identifier=id,
        file__project=project
    )

    if queryset.exists():
        unit = queryset.first()
        print("Found:", unit)
        if unit.description=="CODEFEATURE":
            return None
        content=[{"title": "ID", "content": id}, {"title": "Description", "content": unit.description}]
        # content=[{"title": "ID", "content": id}, {"title": "Description", "content": unit.description}, {"title": "Score", "content": unit.score}, {"title": "Transformed description", "content": unit.transformed_description}]
        return content
    else:
        unit = None
        print("Not found.")
        return None

def get_code_feature_details(uri, project):
    content=[{"title": "ID", "content": uri}]
    query = get_node_description_sparql(uri)
    results = query_graphdb(query, project.name)
    data = results[0] if results else {}
    type=data.get("type", {}).get("value", "")
    level=data.get("level", {}).get("value", "")
    if type!="":
        if level!="":
            content.append({"title": "Type", "content": type.split("#")[1]+" (Level "+level+")"})
        else:
            content.append({"title": "Type", "content": type.split("#")[1]})
    score=data.get("score", {}).get("value", "")
    if score!="":
        content.append({"title": "Score", "content": score})
    desc=data.get("description", {}).get("value", "")
    if desc!="":
        content.append({"title": "Description", "content": desc})
    code=data.get("code", {}).get("value", "")
    if code!="":
        content.append({"title": "Code", "content": code})
    return content

def get_source_details(request, project):
    uri = request.GET.get('uri', '')
    parsed = urlparse(uri)
    fragment = parsed.fragment or uri.split("#")[-1]
    _,_,vector_collection_requirements = get_vector_collection_names(project.name)

    content=[]
    content = get_artifact_details(project, fragment)
    if content==None:
        content = get_code_feature_details(uri, project)
    return fragment, content
