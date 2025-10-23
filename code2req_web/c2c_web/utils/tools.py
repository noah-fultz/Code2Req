from c2c_web.utils.utils import query_vector_db, embed


def get_software_requirement(query, vector_collection, top_k):
    embedded_query=embed(query)[0]
    vector_response = query_vector_db(embedded_query, vector_collection, top_n=top_k, include=["distances", "metadatas", "documents"])
    print("Used RAG, got "+str(len(vector_response["documents"]))+" documents")
    return vector_response["documents"]
