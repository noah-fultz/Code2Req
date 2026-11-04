from pathlib import Path
from c2c_web.utils.utils import get_exists, get_code_artifacts_for_file
from rdflib import URIRef, RDF

preproc_count = 0
global_var_count = 0

# TODO: extract_code_artifacts_c
'''
* Takes in the source code from a C source or header file both raw and parsed into
* a Syntax tree, and returns a set of artifacs/imports extracted from the source
* @param tree - The C Syntax Tree parsed from `code_bytes`
* @param code_bytes - The byte-digest of the Java source
'''
def extract_code_artifacts_c(tree, code_bytes):
    ''' Converts a slice of code bytes back into their plaintext form '''
    def node_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode()

    ''' Discovers the name of a node from the syntax-tree child, defaults to "unknown" '''
    def get_name(node):
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return node_text(name_node)

        # Iterate through the node list until an identifier is found
        curr_declarator = node.child_by_field_name("declarator")
        while curr_declarator is not None:
            if curr_declarator.type == "identifier":
                return node_text(curr_declarator)
            curr_declarator = curr_declarator.child_by_field_name("declarator")
        # If no identifier is found, return unknown
        return "unknown"

    '''
    * Recursively walks through a node's children to discover artifacts defined within
    * them
    * @param parent_node - The node being searched for child artifact definitions
    * @param seen_nodes - The list of nodes already discovered in previous recursions
    * @param scope_name - The name of the scope that contains the artifact
    '''
    def extract_inner_artifacts(parent_node, seen_nodes, scope_name='global', scope_snippet_index=0):
        # Holds the inner artifacts and imports
        inner_artifacts = []
        inner_imports = []

        for child in parent_node.children:
            # If the child is an identifier, ignore it
            if child == parent_node.child_by_field_name("name"):
                continue
            # If the child is directly one of the preprocessing directives, ignore it
            elif child.type in ("#ifdef", "#ifndef", "#endif", ";"):
                continue
            # If the child node has already been seen, ignore it
            elif child in seen_nodes:
                continue
            # Otherwise, add it to the list of seen nodes
            else:
                seen_nodes.add(child)

            
            # Process discovered imports
            if child.type == "preproc_include":
                text = node_text(child)
                inner_imports.append(text)

            # Process function definitions
            elif child.type == "function_definition":
                # Recover the name of the function
                function_name = get_name(child).strip()
                # Push the function into the artifact list
                inner_artifacts.append(("Function", node_text(child), function_name))

                # TODO: Add recursive processing of function definitions

            # Process preprocessing directives
            elif child.type == "preproc_def":
                # Create a globally unique name for the preprocessing directive
                preproc_name = f"preproc_{scope_name}_{scope_snippet_index}"
                scope_snippet_index += 1
                # Push the preprocessing directive as a snippet to the artifact list
                inner_artifacts.append(("Snippet", node_text(child), preproc_name))

                # TODO: Handle function-like macros (e.g. #define f(x) ...)

            # Process globally-scoped variable defintiions
            elif child.type == "declaration":
                # Create a globally unique name for the preprocessing directive
                snippet_name = f"declaration_{scope_name}_{scope_snippet_index}"
                scope_snippet_index += 1
                # Push the declaration as a snippet to the artifact list
                inner_artifacts.append(("Snippet", node_text(child), snippet_name))

            # Process ifdef and ifndef
            elif child.type == "preproc_ifdef":
                # Recurse into the body of the ifdef/ifndef statement
                child_artifacts, child_imports, scope_snippet_index = extract_inner_artifacts(child, seen_nodes, scope_name, scope_snippet_index)
                # Aggregate the artifacts and imports from the recursion
                inner_artifacts += child_artifacts
                inner_imports += child_imports

            # Process struct definitions
            elif child.type == "struct_specifier":
                # Pull the name of the struct
                struct_name = get_name(child).strip()
                # Push the function into the artifact list
                inner_artifacts.append(("Class", node_text(child), struct_name))

            # Ignore comment lines
            elif child.type == "comment":
                pass

            # Fail explicitly on unrecognized node types
            else:
                err = f"Unknown node type '{child.type}' encountered"
                print(f"{err} => {child}")
                print("Source:")
                print(f"{node_text(child)}")
                raise TypeError(err)

        return inner_artifacts, inner_imports, scope_snippet_index

    # Main functionality of this extractor
    root = tree.root_node
    seen_nodes = set()  # Intentionally present, currently unused

    artifacts, imports, _ = extract_inner_artifacts(root, seen_nodes)

    if imports:
        import_block = "\n".join(imports)
        artifacts.insert(0, ("Imports", import_block, "imports"))
    # Return the extracted artifacts
    return artifacts


def resolve_import_to_path_c(import_path, file_uri, project_root, BASE_URI):
    base_path = "/".join(file_uri.replace(f"{BASE_URI}/", "").split("/")[:-1])
    return f"{project_root}/{base_path}/{import_path}"


'''
* Builds or extends a lookup table that matches names to C artifacts
* @param root - The root of the import block's syntax tree
* @param code - The byte-digested aggregated import block
* @param graph - The repository relationship graph
* @param import_references - A mutable dictionary to be initialized or extended
* @param REPO_ROOT_PATH - The root of the cached repository
* @param BASE_URI - The URI base for the graph-based repo representation
* @param EX - The repo's vocabulary namespace
'''
def extract_from_import_c(file_uri, root, code, graph, import_references, REPO_ROOT_PATH, BASE_URI, EX):
    def node_text(node):
        return code[node.start_byte:node.end_byte].decode()

    # Iterate through the top-level children of the import block
    for node in root.children:
        # Ignore all non-include declarations
        if node.type != "preproc_include":
            continue

        path_node = node.child_by_field_name("path")
        # Check whether it's a quoted or bracketed import
        is_quoted = path_node.type == "string_literal"

        # TODO: Handle quoted imports
        if is_quoted:
            import_text = node_text(path_node).strip("\n\t\"")
            print(f"import {import_text} is quoted")
            recovered_path = resolve_import_to_path_c(import_text, file_uri, REPO_ROOT_PATH, BASE_URI)
            print(f"  => resolved to \"{recovered_path}\"")
        # TODO: Handle bracketed imports
        else:
            print(f"import {node_text(path_node).strip()} is not quoted")
        
        print()

    return import_references


# TODO: add_references_c