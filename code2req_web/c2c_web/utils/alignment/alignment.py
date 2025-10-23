
from c2c_web.models import EvaluationFile, EvaluationUnit 
from c2c_web.utils.utils import query_graphdb
from rdflib import URIRef, RDF
from typing import List
import os

name_map={"Code_Features": "From Feature", "Req_from_code":"From Code", "Code_Features_RAG":"From Feature With RAG", "Req_from_code_RAG": "From Code With RAG"}

def get_alignment(project):
    alignments=[]
    url_top_k=None
    url_top_perc=None
    url_f1=None
    url_perc_rec=None
    
    if project.name=="NoProjectSelected":
        return alignments, url_top_k, url_top_perc, url_f1, url_perc_rec
    
    for eval_file in project.evaluation_files.all():
        name=eval_file.name
        if name in name_map:
            name=name_map[name]
        alignment={"name":name, "id":name.replace(" ","_"), "f1":eval_file.f1_score, "total":eval_file.num_units, "linked":eval_file.num_linked_to_feature, "alignments":[]}
        eval_units = EvaluationUnit.objects.filter(file=eval_file)
        for unit in eval_units:
            linked="Yes"
            if unit.linked_feature_ids==[]:
                linked="No"
            linked_classes=set()
            for f in unit.linked_feature_ids:
                related_unit = EvaluationUnit.objects.filter(identifier=f).first()
                for rc in related_unit.linked_code:
                    linked_classes.add(rc)
            alignment["alignments"].append({"source":unit.identifier, "linked":linked, "scores":unit.scores, "linked_sources":unit.linked_feature_ids, "linked_code":unit.linked_code, "linked_classes":list(linked_classes)})
        alignments.append(alignment)

    return alignments
