from pathlib import Path
from tree_sitter import Language, Parser
from rdflib import Graph, Namespace, URIRef, Literal, RDF
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from c2c_web.utils.utils import get_code_artifacts_for_file
from c2c_web.utils.process.process_repo.parse_python import extract_code_artifacts_python, extract_from_import_python, add_references_python
from c2c_web.utils.process.process_repo.parse_java import extract_code_artifacts_java, extract_from_import_java, add_references_java


# === SETUP TREE-SITTER ===
PY_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())

# === NAMESPACES ===
EX = None # to be set dynamically
BASE_URI = None # to be set dynamically
REPO_ROOT_PATH = None  # to be set dynamically
lines_of_code=0

# === RDF GRAPH ===
def create_rdf_graph():
    g = Graph()
    g.bind("ex", EX)
    return g

def safe_read_text(file_path):
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding='utf-8-sig')  # handles BOM
        except UnicodeDecodeError:
            return file_path.read_text(encoding='ISO-8859-1')  # or 'cp1252'


# === PROCESS SINGLE FILE ===
def process_file(graph, file_path, dir_uri, lang_prefix, parser):
    relative_path = file_path.relative_to(REPO_ROOT_PATH)
    file_uri = URIRef(f"{BASE_URI}/{relative_path.as_posix()}")
    graph.add((dir_uri, EX.hasFile, file_uri))
    graph.add((file_uri, RDF.type, EX.CodeFile))
    graph.add((file_uri, EX.name, Literal(file_path.name)))
    code = safe_read_text(file_path)
    global lines_of_code
    lines_of_code+=len(code.split("\n"))
    code_bytes = bytes(code, 'utf-8')
    tree = parser.parse(code_bytes)
    if lang_prefix==".py":
        artifacts = extract_code_artifacts_python(tree, code_bytes)
    elif lang_prefix==".java":
        artifacts = extract_code_artifacts_java(tree, code_bytes)

    artifact_uris = []
    for idx, (kind, code_snippet, name) in enumerate(artifacts):
        artifact_name = name if name else f"artifact{idx}"
        artifact_uri = URIRef(f"{file_uri}#{artifact_name}")
        artifact_uris.append((artifact_uri, code_snippet))
        graph.add((file_uri, EX.containsArtifact, artifact_uri))
        graph.add((artifact_uri, RDF.type, EX[kind]))
        graph.add((artifact_uri, EX.code, Literal(code_snippet)))
        if name:
            graph.add((artifact_uri, EX.name, Literal(name)))

# === WALK DIRECTORY ===
def process_directory(graph, dir_path, parent_uri=None):
    relative_path = dir_path.relative_to(REPO_ROOT_PATH)
    dir_uri = URIRef(f"{BASE_URI}/{relative_path.as_posix()}") if relative_path != Path(".") else URIRef(BASE_URI)
    graph.add((dir_uri, RDF.type, EX.CodeDirectory))
    graph.add((dir_uri, EX.name, Literal(dir_path.name)))
    if parent_uri:
        graph.add((parent_uri, EX.hasDirectory, dir_uri))
    for entry in dir_path.iterdir():
        if entry.name == "__MACOSX":
            continue 
        if entry.is_dir():
            process_directory(graph, entry, dir_uri)
        elif entry.suffix == ".py":
            parser = Parser(PY_LANGUAGE)
            process_file(graph, entry, dir_uri, ".py", parser)
        elif entry.suffix == ".java":
            parser = Parser(JAVA_LANGUAGE)
            process_file(graph, entry, dir_uri, ".java", parser)

def get_code(artifact,graph):
    code_literal = graph.value(subject=artifact, predicate=EX.code)
    return code_literal


def process_call_stack(graph, REPO_ROOT_PATH):
    all_code_files=list(graph.subjects(RDF.type, EX.CodeFile))
    for file_uri in all_code_files:
        if file_uri.endswith(".py"):
            parser = Parser(PY_LANGUAGE)
            lang=".py"
        elif file_uri.endswith(".java"):
            parser = Parser(JAVA_LANGUAGE)
            lang=".java"
        import_references={}
        for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
            if artifact.endswith("imports"):
                imports = get_code(artifact,graph)
                tree = parser.parse(imports.encode())
                if lang==".py":
                    import_references = extract_from_import_python(tree.root_node, imports.encode(),graph, import_references, REPO_ROOT_PATH, BASE_URI, EX)
                elif lang==".java":
                    import_references = extract_from_import_java(artifact,tree.root_node, imports.encode(),graph, import_references, REPO_ROOT_PATH, BASE_URI, EX)
            else:
                import_references[artifact.split("#")[-1]]=artifact
        for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
            if artifact.endswith("imports")==False:
                code = get_code(artifact,graph)
                tree = parser.parse(code.encode())
                if lang==".py":
                    add_references_python(tree.root_node, code.encode(), import_references, artifact, graph, EX)
                if lang==".java":
                    add_references_java(tree.root_node, code.encode(), import_references, artifact, graph, EX)

# === MAIN FUNCTION ===
def build_code_repository_graph(repo_root, ex):
    global BASE_URI, REPO_ROOT_PATH, EX
    graph = create_rdf_graph()
    REPO_ROOT_PATH = Path(repo_root).resolve()
    BASE_URI = f"http://example.org/{REPO_ROOT_PATH.name}"
    EX = Namespace(ex+"#")
    repo_uri = URIRef(BASE_URI)
    graph.add((repo_uri, RDF.type, EX.CodeRepository))
    graph.add((repo_uri, EX.name, Literal(REPO_ROOT_PATH.name)))

    process_directory(graph, REPO_ROOT_PATH, repo_uri)
    process_call_stack(graph, REPO_ROOT_PATH)

    return graph, lines_of_code

