
from pathlib import Path
from c2c_web.utils.utils import get_exists, get_code_artifacts_for_file
from rdflib import URIRef

# === EXTRACT CODE ARTIFACTS ===
def extract_code_artifacts_python(tree, code_bytes):
    def node_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode()

    def get_name(node):
        for child in node.children:
            if child.type == "identifier":
                return node_text(child)
        return "unknown"

    root = tree.root_node
    artifacts = []
    imports = []
    snippets = []
    snippetindex=0

    for node in root.children:
        if node.type == "class_definition":
            artifacts.append(("Class", node_text(node), get_name(node)))
        elif node.type == "function_definition":
            artifacts.append(("Function", node_text(node), get_name(node)))
        elif node.type in ("import_statement", "import_from_statement", "future_import_statement"):
            imports.append(node_text(node))
        elif node.type in ("if_statement", "for_statement", "while_statement", "try_statement", "with_statement", "match_statement"):
            snippetindex+=1
            artifacts.append(("Snippet", node_text(node), "snippet_"+str(snippetindex)))
        elif node.type !="comment":
            snippets.append(node_text(node))
    
    if snippets:
        snippet_block = "\n".join(snippets)
        artifacts.insert(0, ("Snippet", snippet_block, "snippet_0"))

    if imports:
        import_block = "\n".join(imports)
        artifacts.insert(0, ("Imports", import_block, "imports"))

    return artifacts

def resolve_import_to_path(import_path: str, project_root: Path, BASE_URI) -> str:
    if import_path.startswith(".."):
        import_path=import_path[2:]
    elif import_path.startswith("."):
        import_path=import_path[1:]
    path = list(project_root.rglob(import_path.replace(".","/")+".py"))
    return_path=""
    for p in path:
        return_path=str(p).replace(str(project_root),"")
    return_path=f"{BASE_URI}{return_path}"
    return return_path

def extract_from_import_python(root, code, graph, import_references, REPO_ROOT_PATH, BASE_URI, EX):
    for node in root.children:
        if node.type=="import_from_statement":
            imp_file=""
            for i, child in enumerate(node.children):
                if (child.type=="dotted_name" or child.type=="relative_import") and i==1:
                    from_import=code[child.start_byte:child.end_byte].decode()
                    n=resolve_import_to_path(from_import,REPO_ROOT_PATH, BASE_URI)
                    if get_exists(n, graph):
                        imp_file=n
                    else:
                        break
                elif child.type=="dotted_name" and i>1:
                    from_import=code[child.start_byte:child.end_byte].decode()
                    n=f"{imp_file}#{from_import}"
                    if get_exists(n, graph):
                        import_references[from_import]=n
                elif child.type=="aliased_import":
                    alias_import=code[child.start_byte:child.end_byte].decode()
                    from_import, alias = alias_import.split(" as ")
                    n=f"{imp_file}#{from_import}"
                    if get_exists(n, graph):
                        import_references[alias]=n
                elif child.type=="wildcard_import":
                    artifacts = get_code_artifacts_for_file(imp_file, graph, EX)
                    for a in artifacts:
                        import_references[a.split("#")[-1]]=a
    return import_references



def add_references_python(node, code, import_references, this, graph, EX):
    for child in node.children:
        add_references_python(child, code, import_references, this, graph, EX)

    # 1. Match raw identifiers (e.g., getPathwayPairs passed to delayed)
    if node.type == "identifier":
        name = code[node.start_byte:node.end_byte].decode()
        if name in import_references and import_references[name] != this:
            graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 2. Match function calls (e.g., getPathwayPairs(...) or delayed(...))
    elif node.type == "call":
        # Get the callable name from the call expression
        called_expr = node.child_by_field_name("function")
        if called_expr and called_expr.type == "identifier":
            name = code[called_expr.start_byte:called_expr.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 3. Match attribute access (e.g., module.func)
    elif node.type == "attribute":
        # Just match the attribute part, e.g., `func` in `module.func`
        attr_name = node.child_by_field_name("attribute")
        if attr_name:
            name = code[attr_name.start_byte:attr_name.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))
