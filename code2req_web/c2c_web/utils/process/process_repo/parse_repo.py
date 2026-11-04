from pathlib import Path
from tree_sitter import Language, Parser
from rdflib import Graph, Namespace, URIRef, Literal, RDF
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
from c2c_web.utils.utils import get_code_artifacts_for_file
from c2c_web.utils.process.process_repo.parse_python import extract_code_artifacts_python, extract_from_import_python, add_references_python
from c2c_web.utils.process.process_repo.parse_java import extract_code_artifacts_java, extract_from_import_java, add_references_java
from c2c_web.utils.process.process_repo.parse_c import extract_code_artifacts_c, extract_from_import_c

# === SETUP TREE-SITTER ===
PY_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())
C_LANGUAGE = Language(tsc.language())

# === NAMESPACES ===
EX = None # to be set dynamically
BASE_URI = None # to be set dynamically
REPO_ROOT_PATH = None  # to be set dynamically
lines_of_code = 0

# === RDF GRAPH ===
'''
* Create a graph bound into the globally defined custom namespace
'''
def create_rdf_graph():
    g = Graph()
    g.bind("ex", EX)
    return g

'''
* Returns a string extracted from a source file that is encoding-aware
'''
def safe_read_text(file_path):
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding='utf-8-sig')  # handles BOM
        except UnicodeDecodeError:
            return file_path.read_text(encoding='ISO-8859-1')  # or 'cp1252'


# === PROCESS SINGLE FILE ===
'''
* Converts the contents of a source code file into graph nodes for analysis
* @param graph - The Graph that the nodes are inserted into
* @param file_path - The source code file being processed
* @param dir_uri - The URI of the source file's parent directory
* @param lang_prefix - The language-identifying file extension
* @param parser - The languate parser to use in analysis
'''
def process_file(graph, file_path, dir_uri, lang_prefix, parser):
    # Convert the file path into a URI that preserves its position relative
    # to the repository root
    relative_path = file_path.relative_to(REPO_ROOT_PATH)
    file_uri = URIRef(f"{BASE_URI}/{relative_path.as_posix()}")
    # Mark the directory's relationship to this file
    graph.add((dir_uri, EX.hasFile, file_uri))
    # Store the file as a node in the graph
    graph.add((file_uri, RDF.type, EX.CodeFile))
    graph.add((file_uri, EX.name, Literal(file_path.name)))

    # Extract a readable copy of the source code
    code = safe_read_text(file_path)
    # Count the lines of code in this file toward the total
    global lines_of_code
    lines_of_code += len(code.split("\n"))
    
    # Convert the code into raw bytes for parsing into an AST
    code_bytes = bytes(code, 'utf-8')
    tree = parser.parse(code_bytes)
    
    # Handle Python source code
    if lang_prefix == ".py":
        artifacts = extract_code_artifacts_python(tree, code_bytes)
    # Handle Java source code
    elif lang_prefix == ".java":
        artifacts = extract_code_artifacts_java(tree, code_bytes)
    # Handle C source and header files
    elif lang_prefix == ".c" or lang_prefix == ".h":
        artifacts = extract_code_artifacts_c(tree, code_bytes)

    artifact_uris = []
    # Iterate over each artifact extracted from the list
    for idx, (kind, code_snippet, name) in enumerate(artifacts):
        # Ensure the artifact has a valid name
        artifact_name = name if name else f"artifact{idx}"
        # Create a URI for the artifact and push it to the list
        artifact_uri = URIRef(f"{file_uri}#{artifact_name}")
        artifact_uris.append((artifact_uri, code_snippet))
        
        # Mark the file's relationship to this artifact
        graph.add((file_uri, EX.containsArtifact, artifact_uri))
        # Store the artifact as a node in the graph
        graph.add((artifact_uri, RDF.type, EX[kind]))
        graph.add((artifact_uri, EX.code, Literal(code_snippet)))
        # If there was a valid name provided to this iteration, give it to the node
        if name:
            graph.add((artifact_uri, EX.name, Literal(name)))

# === WALK DIRECTORY ===
'''
* Recursively scans a repository and converts the language-specific artifacts into
* nodes in the graph for analysis
* @param graph - The Graph that the nodes are inserted into
* @param dir_path - The path of the directory being walked
* @param parent_uri - The URI of this directory's parent; default None
'''
def process_directory(graph, dir_path, parent_uri=None):
    # Convert the directory path into a URI that preserves its position relative
    # to the repository root
    relative_path = dir_path.relative_to(REPO_ROOT_PATH)
    dir_uri = URIRef(f"{BASE_URI}/{relative_path.as_posix()}") if relative_path != Path(".") else URIRef(BASE_URI)
    # Push the directory URI as a code directory with it's name as a string literal to the graph
    graph.add((dir_uri, RDF.type, EX.CodeDirectory))
    graph.add((dir_uri, EX.name, Literal(dir_path.name)))

    # If this is a subdirectory of an existing directory, store the relationship
    if parent_uri:
        graph.add((parent_uri, EX.hasDirectory, dir_uri))

    # Iterate through each entry in the current directory for analysis
    for entry in dir_path.iterdir():
        # If the entry is MacOS-specific data, ignore it
        if entry.name == "__MACOSX":
            continue 
        # If the entry is a directory, recurse into it
        if entry.is_dir():
            process_directory(graph, entry, dir_uri)
        # If the entry is a Python file, parse it using the Python file processor
        elif entry.suffix == ".py":
            parser = Parser(PY_LANGUAGE)
            process_file(graph, entry, dir_uri, ".py", parser)
        # If the entry is a Java file, parse it using the Java file processor
        elif entry.suffix == ".java":
            parser = Parser(JAVA_LANGUAGE)
            process_file(graph, entry, dir_uri, ".java", parser)
        # If the entry is a C source or header fiel, parse it using the C file processor
        elif entry.suffix == ".c" or entry.suffix == ".h":
            parser = Parser(C_LANGUAGE)
            process_file(graph, entry, dir_uri, entry.suffix, parser)

'''
* Pulls the source code associated with an artifact
* @param artifact - The artifact whose source code is being pulled
* @param graph - The Graph that the nodes are read from
'''
def get_code(artifact, graph):
    code_literal = graph.value(subject=artifact, predicate=EX.code)
    return code_literal


'''
* Passes through the list of files and artifacts in the graph and resolves
* their imports. Adds missing relationships between imported artifacts
* @param graph - The Graph that the nodes are inserted into
* @param REPO_ROOT_PATH - The absolute(?) path to the repository's root
'''
def process_call_stack(graph, REPO_ROOT_PATH):
    # Enumerate a list of all code files in the graph
    all_code_files = list(graph.subjects(RDF.type, EX.CodeFile))
    # Iterate through each file in the enumerated list
    for file_uri in all_code_files:
        # If the file is a Python file, set the proper parser and extension 
        if file_uri.endswith(".py"):
            parser = Parser(PY_LANGUAGE)
            lang = "python"
        # If the file is a Java file, set the proper parser and extension
        elif file_uri.endswith(".java"):
            parser = Parser(JAVA_LANGUAGE)
            lang = "java"
        # If the file is a C source or header file, set the proper parser and extension
        elif file_uri.endswith(".c") or file_uri.endswith(".h"):
            parser = Parser(C_LANGUAGE)
            lang = "c"

        import_references = {}
        # Iterate through each artifact in the file and build a named artifact lookup
        for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
            # Check if the artifact is the import block
            if artifact.endswith("imports"):
                # Pull and parse the plaintext of the import block
                imports = get_code(artifact, graph)
                tree = parser.parse(imports.encode())
                
                # If it's Python, dispatch the import resolution to the Python extractor
                if lang == "python":
                    import_references = extract_from_import_python(tree.root_node, imports.encode(), graph, import_references, REPO_ROOT_PATH, BASE_URI, EX)
                # If it's Java, dispatch the import resolution to the Java extractor
                elif lang == "java":
                    import_references = extract_from_import_java(artifact, tree.root_node, imports.encode(), graph, import_references, REPO_ROOT_PATH, BASE_URI, EX)
                # If it's C, dispatch the import resolution to the C extractor
                elif lang == "c":
                    print("Processing imports")
                    print("============================================================")
                    import_references = extract_from_import_c(file_uri, tree.root_node, imports.encode(), graph, import_references, REPO_ROOT_PATH, BASE_URI, EX)
                    print("============================================================\n")

            # If the artifact is not an import, store it as a named artifact for reference in later import checks
            else:
                import_references[artifact.split("#")[-1]] = artifact
        
        # Parse each non-import artifact and use the lookup to resolve relationships
        for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
            # Ensure this is not an import
            if artifact.endswith("imports") == False:
                # Pull and parse the artifact's source
                code = get_code(artifact, graph)
                tree = parser.parse(code.encode())
                
                # If it's Python, dispatch reference resolution
                if lang == "python":
                    add_references_python(tree.root_node, code.encode(), import_references, artifact, graph, EX)
                # If it's Java, dispatch reference resolution
                elif lang == "java":
                    add_references_java(tree.root_node, code.encode(), import_references, artifact, graph, EX)
                # TODO: Add C processing
                elif lang == "c":
                    raise ValueError("C-language file encountered")



# === MAIN FUNCTION ===
'''
* Main entrypoint for repository analysis
* @param repo_root - The path to the root of the repository being analyzed
* @param ex - The provided namespace to be used by other methods
'''
def build_code_repository_graph(repo_root, ex):
    # Creates a shared state among the analyzers, these global variables
    # can be accessed and used by other methods
    # TODO: Check if this clobbers data when multiple repos are analyzed at once
    global BASE_URI, REPO_ROOT_PATH, EX
    
    # Creates a graph to retain the relationships between nodes
    graph = create_rdf_graph()
    # Constructs a URI for the resources using an absolutized repository path
    REPO_ROOT_PATH = Path(repo_root).resolve()
    BASE_URI = f"http://example.org/{REPO_ROOT_PATH.name}"
    # Defines the global vocabulary namespace
    EX = Namespace(ex+"#")
    # Wraps the repository URI as a Term
    repo_uri = URIRef(BASE_URI)
    # Adds the repository as a resource and pushes the name as a string literal into the graph
    graph.add((repo_uri, RDF.type, EX.CodeRepository))
    graph.add((repo_uri, EX.name, Literal(REPO_ROOT_PATH.name)))

    # Recursively scan the repository and parse the language-specific artifacts into graph nodes
    process_directory(graph, REPO_ROOT_PATH, repo_uri)
    # Make a second pass over the artifacts and resolve imports via links
    process_call_stack(graph, REPO_ROOT_PATH)

    # Returns the completed graph and number of osurce lines counted during processing
    return graph, lines_of_code

