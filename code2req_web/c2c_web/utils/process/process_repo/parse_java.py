from pathlib import Path
from c2c_web.utils.utils import get_exists, get_code_artifacts_for_file
from rdflib import URIRef, RDF

def extract_code_artifacts_java(tree, code_bytes):
    def node_text(node):
        return code_bytes[node.start_byte:node.end_byte].decode()

    def get_name(node):
        name_node = node.child_by_field_name("name")
        return node_text(name_node) if name_node else "unknown"

    def node_text_with_docstring(node):
        doc = ""
        prev = node.prev_sibling
        while prev and prev.type == "comment":
            text = code_bytes[prev.start_byte:prev.end_byte].decode().strip()
            if text.startswith("/**"):
                doc = text + "\n" + doc
                break
            prev = prev.prev_sibling
        return doc + code_bytes[node.start_byte:node.end_byte].decode()

    def extract_inner_artifacts(parent_node, seen_nodes):
        inner = []
        snippet_index = 0

        def walk(n):
            nonlocal snippet_index

            if n in seen_nodes:
                return

            if n.type == "method_declaration":
                seen_nodes.add(n)
                inner.append(("Function", node_text_with_docstring(n), get_name(n), n))

            elif n.type == "class_declaration":
                #  Don't add top-level class again — check for nesting
                if n != parent_node:
                    seen_nodes.add(n)
                    inner.append(("Class", node_text_with_docstring(n), get_name(n), n))
                    walk_children(n)  # recurse into inner class

            elif n.type in (
                "if_statement", "for_statement", "while_statement",
                "try_statement", "switch_expression", "synchronized_statement"
            ):
                seen_nodes.add(n)
                snippet_index += 1
                name = f"snippet_{snippet_index}"
                inner.append(("Snippet", node_text(n), name, n))

            else:
                walk_children(n)

        def walk_children(n):
            for child in n.children:
                walk(child)

        walk_children(parent_node)
        return inner


    def build_class_stub_with_removal(class_node, child_artifacts, code_bytes):
        def node_text(n):
            return code_bytes[n.start_byte:n.end_byte].decode()

        def get_name(node):
            name_node = node.child_by_field_name("name")
            return node_text(name_node) if name_node else "unknown"

        header = ""
        for child in class_node.children:
            if child.type in ("modifiers", "class", "interface", "enum"):
                header += node_text(child) + " "
            elif child.type == "identifier":
                header += node_text(child)
            elif child.type in ("type_parameters", "superclass", "super_interfaces"):
                header += " " + node_text(child)
        header = header.strip() + " {"

        body_lines = []

        # Find the actual class body node
        class_body_node = next((c for c in class_node.children if c.type == "class_body"), None)
        if class_body_node:
            for element in class_body_node.children:
                if element.type in ("method_declaration", "class_declaration"):
                    # replaced by stub
                    for kind, _, name, artifact_node in child_artifacts:
                        if artifact_node == element:
                            if kind == "Function":
                                body_lines.append(f"    void {name}();  // artifact reference")
                            elif kind == "Class":
                                body_lines.append(f"    class {name};  // inner class reference")
                            break
                elif element.type not in ("}", "{"):  # ignore braces
                    body_lines.append("    " + node_text(element).strip())

        return header + "\n" + "\n".join(body_lines) + "\n}"


    # === MAIN ===
    root = tree.root_node
    artifacts = []
    imports = []
    seen_nodes = set()

    for node in root.children:
        # print(node.type)
        if node.type == "import_declaration":
            imports.append(code_bytes[node.start_byte:node.end_byte].decode())

        elif node.type == "class_declaration" and node not in seen_nodes:
            seen_nodes.add(node)
            class_name = get_name(node)

            # Extract inner methods/snippets/classes and prevent duplicates
            inner_artifacts = extract_inner_artifacts(node, seen_nodes)

            # Build stub for class referencing children
            stub_code = build_class_stub_with_removal(node, inner_artifacts, code_bytes)
            artifacts.append(("Class", stub_code, class_name))

            # Add inner artifacts separately
            for kind, code, name, _ in inner_artifacts:
                artifacts.append((kind, code, name))

    if imports:
        import_block = "\n".join(imports)
        artifacts.insert(0, ("Imports", import_block, "imports"))

    return artifacts





def resolve_import_to_path_java(import_path: str, project_root: Path, BASE_URI: str) -> str:
    # Convert package-like import to path
    import_path = import_path.replace(".", "/")
    matches = list(project_root.rglob(f"{import_path}.java"))
    return_path = ""
    for match in matches:
        if "__MACOSX" not in str(match):
            return_path = str(match).replace(str(project_root), "")
    
    return f"{BASE_URI}{return_path}"

def extract_from_import_java(artifact, root, code, graph, import_references, REPO_ROOT_PATH, BASE_URI, EX):
    for node in root.children:
        if node.type != "import_declaration":
            continue

        import_text = code[node.start_byte:node.end_byte].decode().strip().rstrip(";")
        is_static = "static " in import_text
        import_path = import_text.replace("import", "").replace("static", "").strip()

        # --- 1. Static wildcard import ---
        if is_static and import_path.endswith(".*"):
            class_path = import_path[:-2]
            resolved_path = resolve_import_to_path_java(class_path, REPO_ROOT_PATH, BASE_URI)
            if resolved_path:
                for artifact in get_code_artifacts_for_file(resolved_path, graph, EX):
                    class_name = graph.value(artifact, EX.name)
                    if class_name:
                        import_references[str(class_name)] = artifact

        # --- 2. Static single member import ---
        elif is_static:
            parts = import_path.split(".")
            class_name = parts[-2]
            member_name = parts[-1]
            class_path = ".".join(parts[:-1])
            resolved_path = resolve_import_to_path_java(class_path, REPO_ROOT_PATH, BASE_URI)
            if resolved_path:
                class_uri = f"{resolved_path}#{class_name}"
                if get_exists(class_uri, graph):
                    import_references[member_name] = class_uri

        # --- 3. Package wildcard import (non-static) ---
        elif import_path.endswith(".*"):
            package_path = import_path[:-2]
            resolved_path = resolve_import_to_path_java(package_path + ".DUMMY", REPO_ROOT_PATH, BASE_URI)
            if resolved_path:
                base_dir = resolved_path.rsplit("/", 1)[0]
                for file_uri in graph.subjects(RDF.type, EX.CodeFile):
                    if str(file_uri).startswith(base_dir):
                        for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
                            name = graph.value(artifact, EX.name)
                            if name:
                                import_references[str(name)] = artifact

        # --- 4. Single class import (non-static) ---
        else:
            resolved_path = resolve_import_to_path_java(import_path, REPO_ROOT_PATH, BASE_URI)
            if resolved_path:
                class_name = import_path.split(".")[-1]
                artifact_uri = f"{resolved_path}#{class_name}"
                if get_exists(artifact_uri, graph):
                    import_references[class_name] = artifact_uri

    # --- 5. Add implicit imports from same package ---
    # Try to find which file this code belongs to
    package_prefix="/".join(artifact.split("/")[:-1])
    for file_uri in graph.subjects(RDF.type, EX.CodeFile):
        if str(file_uri).startswith(package_prefix):
            for artifact in get_code_artifacts_for_file(file_uri, graph, EX):
                name = graph.value(artifact, EX.name)
                if name:
                    import_references[str(name)] = artifact

    return import_references


def add_references_java(node, code, import_references, this, graph, EX):
    for child in node.children:
        add_references_java(child, code, import_references, this, graph, EX)

        # Handle parent class/interface references
    if node.type == "class_declaration":
        for child in node.children:
            if child.type in ("superclass", "super_interfaces"):
                for id_node in child.children:
                    if id_node.type == "identifier":
                        name = code[id_node.start_byte:id_node.end_byte].decode()
                        if name in import_references and import_references[name] != this:
                            graph.add((URIRef(this), EX.extendsClass, URIRef(import_references[name])))

    # 1. Match simple identifier (e.g., a reference to a static class name, field, or method)
    if node.type == "identifier":
        name = code[node.start_byte:node.end_byte].decode()
        if name in import_references and import_references[name] != this:
            graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 2. Match constructor / class instantiation: `new ClassName(...)`
    elif node.type == "object_creation_expression":
        type_node = node.child_by_field_name("type")
        if type_node:
            name = code[type_node.start_byte:type_node.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 3. Match static method calls: `ClassName.staticMethod(...)`
    elif node.type == "method_invocation":
        scope_node = node.child_by_field_name("object")
        if scope_node:
            name = code[scope_node.start_byte:scope_node.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))
        # fallback to raw method name
        method_node = node.child_by_field_name("name")
        if method_node:
            name = code[method_node.start_byte:method_node.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 4. Static field access: `SomeClass.CONSTANT`
    elif node.type == "field_access":
        scope_node = node.child_by_field_name("object")
        if scope_node:
            name = code[scope_node.start_byte:scope_node.end_byte].decode()
            if name in import_references and import_references[name] != this:
                graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[name])))

    # 5. Qualified type references (e.g., `ClassName var = ...`)
    elif node.type in {"type_identifier", "scoped_type_identifier"}:
        name = code[node.start_byte:node.end_byte].decode()
        base = name.split(".")[0]
        if base in import_references and import_references[base] != this:
            graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[base])))

    # 6. Qualified name used as field (e.g., ClassName.FIELD.subField)
    elif node.type == "qualified_identifier":
        base = code[node.start_byte:node.end_byte].decode().split(".")[0]
        if base in import_references and import_references[base] != this:
            graph.add((URIRef(this), EX.usesArtifact, URIRef(import_references[base])))
