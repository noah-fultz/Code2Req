import re
from typing import Dict

from c2c_web.models import EvaluationFile, EvaluationUnit 
from c2c_web.utils.utils import embed, cosine_similarity, get_all_from_vectordb, query_graphdb
import statistics
import pandas as pd
import matplotlib.pyplot as plt

def f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def extract_name(value):
    match = re.search(r'#(.*?)(?:_[a-f0-9]{10,}|)_feature$', value)
    if match:
        ex = match.group(1)
        return ex.replace("_"," ")

def get_embeddings(text_dict):
    """Embeds all texts in a dictionary once and returns a dict of embeddings."""
    embedded_dict={}
    for id in text_dict:
        embedded_dict[id]=embed(text_dict[id])[0]
    return embedded_dict


def get_unit_dict(eval_file):
    eval_units = EvaluationUnit.objects.filter(file=eval_file)
    text_dict={}
    for unit in eval_units:
        # text_dict[unit.identifier]=unit.transformed_description
        text_dict[unit.identifier]=unit.description
    return text_dict


def compare(project, comparison_report_path, similarity_cutoff=0.8):
    code_feature_file = EvaluationFile.objects.filter(project=project, name="Code_Features").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    all_code_features={}
    for feat_req in all_code_feature_reqs:
        all_code_features[feat_req.identifier]=feat_req.description
    embedded_cf = get_embeddings(all_code_features) 
    all_linked_code_features={}

    re_from_code_file = EvaluationFile.objects.filter(project=project, name="Req_from_code").first()
    all_reqs_from_code = EvaluationUnit.objects.filter(file=re_from_code_file)
    all_req_f_code={}
    for feat_req in all_reqs_from_code:
        all_req_f_code[feat_req.identifier]=feat_req.description
    embedded_rc = get_embeddings(all_req_f_code) 
    all_linked_req_from_code={}

    code_feature_rag_file = EvaluationFile.objects.filter(project=project, name="Code_Features_RAG").first()
    all_code_feature_rag_reqs = EvaluationUnit.objects.filter(file=code_feature_rag_file)
    all_code_features_rag={}
    for feat_req in all_code_feature_rag_reqs:
        all_code_features_rag[feat_req.identifier]=feat_req.description
    embedded_cf_rag = get_embeddings(all_code_features_rag) 
    all_linked_code_features_rag={}

    re_from_code_rag_file = EvaluationFile.objects.filter(project=project, name="Req_from_code_RAG").first()
    all_reqs_from_code_rag = EvaluationUnit.objects.filter(file=re_from_code_rag_file)
    all_req_f_code_rag={}
    for feat_req in all_reqs_from_code_rag:
        all_req_f_code_rag[feat_req.identifier]=feat_req.description
    embedded_rc_rag = get_embeddings(all_req_f_code_rag) 
    all_linked_req_from_code_rag={}

    total_count=0
    used_unit_for_feature=set()
    used_unit_for_code=set()
    used_unit_for_feature_rag=set()
    used_unit_for_code_rag=set()
    all_units=0
    for eval_file in project.evaluation_files.all():
        if eval_file.name!="Code_Features" and eval_file.name!="Req_from_code" and eval_file.name!="Code_Features_RAG" and eval_file.name!="Req_from_code_RAG":
            unit_dict=get_unit_dict(eval_file)
            embedded_units = get_embeddings(unit_dict)
            all_units+=len(unit_dict)
            used_code_features=set()
            used_req_from_code=set()
            used_code_features_rag=set()
            used_req_from_code_rag=set()
            all_cf_similarity=[]
            all_rc_similarity=[]
            all_cf_similarity_with_ids=[]
            all_rc_similarity_with_ids=[]
            all_cf_rag_similarity=[]
            all_rc_rag_similarity=[]
            all_cf_rag_similarity_with_ids=[]
            all_rc_rag_similarity_with_ids=[]
            for unit in unit_dict:
                eval_unit = EvaluationUnit.objects.get(identifier=unit, file=eval_file)
                eval_unit.linked_feature_ids=[]
                eval_unit.scores=[]
                eval_unit.save()
                sim_list_cf=[]
                sim_list_rc=[]
                sim_list_cf_with_ids={}
                sim_list_rc_with_ids={}
                sim_list_cf_rag=[]
                sim_list_rc_rag=[]
                sim_list_cf_rag_with_ids={}
                sim_list_rc_rag_with_ids={}
                for cf in all_code_features:
                    similarity=cosine_similarity(embedded_cf[cf], embedded_units[unit])
                    sim_list_cf.append(similarity)
                    sim_list_cf_with_ids[cf]=similarity
                    if similarity>=similarity_cutoff:
                        if cf not in all_linked_code_features:
                            all_linked_code_features[cf]=[]
                        all_linked_code_features[cf].append(unit+":"+f"{similarity:.2f}")
                        
                        used_code_features.add(cf)
                        used_unit_for_feature.add(unit)
                        eval_unit.linked_feature_ids.append(cf)
                        eval_unit.scores.append(f"{similarity:.2f}")
                        eval_unit.save()
                        code_unit = EvaluationUnit.objects.get(identifier=cf, file=code_feature_file)
                        code_unit.linked_feature_ids.append(unit)
                        code_unit.scores.append(f"{similarity:.2f}")
                        code_unit.save()
                for rc in all_req_f_code:
                    similarity=cosine_similarity(embedded_rc[rc], embedded_units[unit])
                    sim_list_rc.append(similarity)
                    sim_list_rc_with_ids[rc]=similarity
                    if similarity>=similarity_cutoff:
                        if rc not in all_linked_req_from_code:
                            all_linked_req_from_code[rc]=[]
                        all_linked_req_from_code[rc].append(unit+":"+f"{similarity:.2f}")
                        
                        used_req_from_code.add(rc)
                        used_unit_for_code.add(unit)
                        eval_unit.linked_feature_ids.append(rc)
                        eval_unit.scores.append(f"{similarity:.2f}")
                        eval_unit.save()
                        code_unit = EvaluationUnit.objects.get(identifier=rc, file=re_from_code_file)
                        code_unit.linked_feature_ids.append(unit)
                        code_unit.scores.append(f"{similarity:.2f}")
                        code_unit.save()
                for cf in all_code_features_rag:
                    similarity=cosine_similarity(embedded_cf_rag[cf], embedded_units[unit])
                    sim_list_cf_rag.append(similarity)
                    sim_list_cf_rag_with_ids[cf]=similarity
                    if similarity>=similarity_cutoff:
                        if cf not in all_linked_code_features_rag:
                            all_linked_code_features_rag[cf]=[]
                        all_linked_code_features_rag[cf].append(unit+":"+f"{similarity:.2f}")
                        
                        used_code_features_rag.add(cf)
                        used_unit_for_feature_rag.add(unit)
                        eval_unit.linked_feature_ids.append(cf)
                        eval_unit.scores.append(f"{similarity:.2f}")
                        eval_unit.save()
                        code_unit = EvaluationUnit.objects.get(identifier=cf, file=code_feature_rag_file)
                        code_unit.linked_feature_ids.append(unit)
                        code_unit.scores.append(f"{similarity:.2f}")
                        code_unit.save()
                for rc in all_req_f_code_rag:
                    similarity=cosine_similarity(embedded_rc_rag[rc], embedded_units[unit])
                    sim_list_rc_rag.append(similarity)
                    sim_list_rc_rag_with_ids[rc]=similarity
                    if similarity>=similarity_cutoff:
                        if rc not in all_linked_req_from_code_rag:
                            all_linked_req_from_code_rag[rc]=[]
                        all_linked_req_from_code_rag[rc].append(unit+":"+f"{similarity:.2f}")
                        
                        used_req_from_code_rag.add(rc)
                        used_unit_for_code_rag.add(unit)
                        eval_unit.linked_feature_ids.append(rc)
                        eval_unit.scores.append(f"{similarity:.2f}")
                        eval_unit.save()
                        code_unit = EvaluationUnit.objects.get(identifier=rc, file=re_from_code_rag_file)
                        code_unit.linked_feature_ids.append(unit)
                        code_unit.scores.append(f"{similarity:.2f}")
                        code_unit.save()
                sorted_sim_list_cf = sorted(sim_list_cf, reverse=True)
                sorted_sim_list_rc = sorted(sim_list_rc, reverse=True)
                all_cf_similarity.append(sorted_sim_list_cf)
                all_rc_similarity.append(sorted_sim_list_rc)
                all_cf_similarity_with_ids.append(sim_list_cf_with_ids)
                all_rc_similarity_with_ids.append(sim_list_rc_with_ids)
                sorted_sim_list_cf_rag = sorted(sim_list_cf_rag, reverse=True)
                sorted_sim_list_rc_rag = sorted(sim_list_rc_rag, reverse=True)
                all_cf_rag_similarity.append(sorted_sim_list_cf_rag)
                all_rc_rag_similarity.append(sorted_sim_list_rc_rag)
                all_cf_rag_similarity_with_ids.append(sim_list_cf_rag_with_ids)
                all_rc_rag_similarity_with_ids.append(sim_list_rc_rag_with_ids)

            count = EvaluationUnit.objects.filter(linked_feature_ids=[], file=eval_file).count()
            total_count+=count
            f1, _, _ = f1_score((len(used_code_features)+len(used_req_from_code)+len(used_code_features_rag)+len(used_req_from_code_rag)), (len(all_code_features)+len(all_req_f_code)+len(all_code_features_rag)+len(all_req_f_code_rag))- (len(used_code_features)+len(used_req_from_code)+len(all_code_features)+len(all_req_f_code)+len(all_code_features_rag)+len(all_req_f_code_rag))- (len(used_code_features_rag)+len(used_req_from_code_rag)), count)
            eval_file.f1_score=f1
            eval_file.num_linked_to_feature=len(unit_dict)-count
            eval_file.num_units=len(unit_dict)
            eval_file.save()

    f1, _, _ = f1_score(len(all_linked_code_features), len(all_code_features) - len(all_linked_code_features), all_units-len(used_unit_for_feature))
    existing_file = EvaluationFile.objects.filter(project=project, name="Code_Features").first()
    existing_file.f1_score=f1
    existing_file.num_linked_to_feature=len(all_linked_code_features)
    existing_file.num_units=len(all_code_features)
    existing_file.save()
    
    code_feature_file = EvaluationFile.objects.filter(project=project, name="Code_Features").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    generated_reqs_file = open(comparison_report_path+"generated_reqs_features.tsv", "w")
    generated_reqs_file.write("#id\tRequirement\tlinked\n")
    for feat_req in all_code_feature_reqs:
        generated_reqs_file.write(feat_req.identifier+"\t"+feat_req.description.replace("\n"," ")+"\t"+str(feat_req.linked_feature_ids)+"\n")
    generated_reqs_file.close()

    f1, _, _ = f1_score(len(all_linked_req_from_code), len(all_req_f_code) - len(all_linked_req_from_code), all_units-len(used_unit_for_code))
    existing_file = EvaluationFile.objects.filter(project=project, name="Req_from_code").first()
    existing_file.f1_score=f1
    existing_file.num_linked_to_feature=len(all_linked_req_from_code)
    existing_file.num_units=len(all_req_f_code)
    existing_file.save()

    code_feature_file = EvaluationFile.objects.filter(project=project, name="Req_from_code").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    generated_reqs_file = open(comparison_report_path+"generated_reqs_code.tsv", "w")
    generated_reqs_file.write("#id\tRequirement\tlinked\n")
    for feat_req in all_code_feature_reqs:
        generated_reqs_file.write(feat_req.identifier+"\t"+feat_req.description.replace("\n"," ")+"\t"+str(feat_req.linked_feature_ids)+"\n")
    generated_reqs_file.close()

    f1, _, _ = f1_score(len(all_linked_code_features_rag), len(all_code_features_rag) - len(all_linked_code_features_rag), all_units-len(used_unit_for_feature_rag))
    existing_file = EvaluationFile.objects.filter(project=project, name="Code_Features_RAG").first()
    existing_file.f1_score=f1
    existing_file.num_linked_to_feature=len(all_linked_code_features_rag)
    existing_file.num_units=len(all_code_features_rag)
    existing_file.save()

    code_feature_file = EvaluationFile.objects.filter(project=project, name="Code_Features_RAG").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    generated_reqs_file = open(comparison_report_path+"generated_reqs_features_rag.tsv", "w")
    generated_reqs_file.write("#id\tRequirement\tlinked\n")
    for feat_req in all_code_feature_reqs:
        generated_reqs_file.write(feat_req.identifier+"\t"+feat_req.description.replace("\n"," ")+"\t"+str(feat_req.linked_feature_ids)+"\n")
    generated_reqs_file.close()
    

    f1, _, _ = f1_score(len(all_linked_req_from_code_rag), len(all_req_f_code_rag) - len(all_linked_req_from_code_rag), all_units-len(used_unit_for_code_rag))
    existing_file = EvaluationFile.objects.filter(project=project, name="Req_from_code_RAG").first()
    existing_file.f1_score=f1
    existing_file.num_linked_to_feature=len(all_linked_req_from_code_rag)
    existing_file.num_units=len(all_req_f_code_rag)
    existing_file.save()

    code_feature_file = EvaluationFile.objects.filter(project=project, name="Req_from_code_RAG").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    generated_reqs_file = open(comparison_report_path+"generated_reqs_code_rag.tsv", "w")
    generated_reqs_file.write("#id\tRequirement\tlinked\n")
    for feat_req in all_code_feature_reqs:
        generated_reqs_file.write(feat_req.identifier+"\t"+feat_req.description.replace("\n"," ")+"\t"+str(feat_req.linked_feature_ids)+"\n")
    generated_reqs_file.close()

    code_feature_file = EvaluationFile.objects.filter(project=project, name="Requirements").first()
    all_code_feature_reqs = EvaluationUnit.objects.filter(file=code_feature_file)
    generated_reqs_file = open(comparison_report_path+"reqs_linked.tsv", "w")
    generated_reqs_file.write("#id\tRequirement\tlinked\n")
    for feat_req in all_code_feature_reqs:
        generated_reqs_file.write(feat_req.identifier+"\t"+feat_req.description.replace("\n"," ")+"\t"+str(feat_req.linked_feature_ids)+"\n")
    generated_reqs_file.close()
    

