from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

from rdflib import RDF, URIRef
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_java
import tree_sitter_python

from c2c_web.utils.process.process_repo.feature_generation import (
    FeaturePromptTooLarge,
    check_feature_prompt_budget,
    get_feature_prompt_budget,
    render_feature_messages,
)


class ArtifactCollectionError(ValueError):
    pass


@dataclass(frozen=True)
class SourceArtifact:
    uri: str
    file_uri: str
    language: str
    kind: str
    name: str
    code: str


@dataclass(frozen=True)
class SourceChunk:
    original_artifact_uri: str
    file_uri: str
    language: str
    part_index: int
    start_byte: int
    end_byte: int
    context: str
    code: str

    @property
    def prompt_key(self):
        if self.part_index == 0:
            return self.original_artifact_uri
        return f"{self.original_artifact_uri}::part-{self.part_index:08d}"


@dataclass(frozen=True)
class SourceBatch:
    chunks: tuple

    @property
    def payload(self):
        return {chunk.prompt_key: chunk.code for chunk in self.chunks}

    @property
    def artifact_uris(self):
        return tuple(sorted({chunk.original_artifact_uri for chunk in self.chunks}))


class SourceSplitError(ValueError):
    pass


def collect_cluster_artifacts(graph, EX, members):
    """Collect full code artifacts and resolve their owning source files."""
    records = []
    for member in sorted({str(uri) for uri in members}):
        artifact_uri = URIRef(member)
        owners = sorted(
            str(owner)
            for owner in graph.subjects(EX.containsArtifact, artifact_uri)
            if (owner, RDF.type, EX.CodeFile) in graph
        )
        if len(owners) != 1:
            raise ArtifactCollectionError(
                f"Expected one owning CodeFile for artifact {member}, found {len(owners)}"
            )

        code_values = sorted({str(value) for value in graph.objects(artifact_uri, EX.code)})
        if len(code_values) != 1:
            raise ArtifactCollectionError(
                f"Expected one code value for artifact {member}, found {len(code_values)}"
            )

        type_names = sorted(
            str(object_uri).rsplit("#", 1)[-1]
            for object_uri in graph.objects(artifact_uri, RDF.type)
            if str(object_uri) != str(EX.Feature)
        )
        kind = type_names[0] if type_names else "Unknown"
        name_values = sorted({str(value) for value in graph.objects(artifact_uri, EX.name)})
        parsed_file_path = urlparse(owners[0]).path
        language = PurePosixPath(parsed_file_path).suffix.lstrip(".").lower() or "unknown"
        records.append(
            SourceArtifact(
                uri=member,
                file_uri=owners[0],
                language=language,
                kind=kind,
                name=name_values[0] if name_values else member.rsplit("#", 1)[-1],
                code=code_values[0],
            )
        )

    return sorted(records, key=lambda record: (record.file_uri, record.uri))


def _parser_for_language(language):
    grammars = {
        "c": tree_sitter_c.language,
        "h": tree_sitter_c.language,
        "java": tree_sitter_java.language,
        "py": tree_sitter_python.language,
    }
    grammar = grammars.get(language)
    if grammar is None:
        raise SourceSplitError(f"No syntax splitter is configured for language '{language}'")
    return Parser(Language(grammar()))


def _parse_source(code, language):
    code_bytes = code.encode("utf-8")
    prefix = b""
    suffix = b""
    parser = _parser_for_language(language)
    tree = parser.parse(code_bytes)
    if language == "java" and tree.root_node.has_error:
        prefix = b"class __Code2ReqChunk { void __code2req__() {\n"
        suffix = b"\n} }"
        tree = parser.parse(prefix + code_bytes + suffix)
    if tree.root_node.has_error:
        raise SourceSplitError(f"Tree-sitter could not parse {language} artifact for splitting")
    return tree.root_node, code_bytes, len(prefix)


def _walk_nodes(node):
    yield node
    for child in node.named_children:
        yield from _walk_nodes(child)


def _body_for_node(node):
    for field in ("body", "consequence", "block", "class_body"):
        body = node.child_by_field_name(field)
        if body is not None and body.named_children:
            return body
    for child in node.named_children:
        if child.type in ("block", "compound_statement", "class_body", "field_declaration_list"):
            if child.named_children:
                return child
    return None


def _find_split_container(root, kind):
    nodes = list(_walk_nodes(root))
    preferred_types = {
        "Function": ("function_definition", "method_declaration", "constructor_declaration"),
        "Class": ("class_definition", "class_declaration", "struct_specifier"),
    }.get(kind, ())
    for preferred_type in preferred_types:
        for node in nodes:
            if node.type == preferred_type:
                body = _body_for_node(node)
                if body is not None:
                    return body
    for node in nodes:
        body = _body_for_node(node)
        if body is not None:
            return body
    return None


def _candidate_fits(code, artifact, feature_type, model, budget, encoding):
    temporary_chunk = SourceChunk(
        original_artifact_uri=artifact.uri,
        file_uri=artifact.file_uri,
        language=artifact.language,
        part_index=1,
        start_byte=0,
        end_byte=len(code.encode("utf-8")),
        context="",
        code=code,
    )
    try:
        check_feature_prompt_budget(
            render_feature_messages({temporary_chunk.prompt_key: code}, feature_type),
            model,
            "source-split",
            budget=budget,
            encoding=encoding,
        )
    except FeaturePromptTooLarge:
        return False
    return True


def split_oversized_artifact(artifact, feature_type, model, budget=None, encoding=None):
    """Split one artifact into syntax-bounded pieces, retaining its source URI."""
    budget = budget or get_feature_prompt_budget()
    fragments = []

    def split_text(text, prefix_context, suffix_context, base_offset, depth):
        if depth > 32:
            raise SourceSplitError(f"Syntax splitting exceeded maximum depth for {artifact.uri}")
        root, source_bytes, wrapper_prefix_length = _parse_source(text, artifact.language)
        body = _find_split_container(root, artifact.kind)
        if body is None:
            raise SourceSplitError(
                f"Oversized artifact {artifact.uri} has no splittable syntax body"
            )

        children = body.named_children
        if not children:
            raise SourceSplitError(
                f"Oversized artifact {artifact.uri} contains no splittable syntax statements"
            )

        child_spans = [
            (child.start_byte - wrapper_prefix_length, child.end_byte - wrapper_prefix_length)
            for child in children
        ]
        if any(start < 0 or end > len(source_bytes) for start, end in child_spans):
            raise SourceSplitError(f"Parser returned invalid source boundaries for {artifact.uri}")

        first_start = child_spans[0][0]
        last_end = child_spans[-1][1]
        fragment_prefix = prefix_context + source_bytes[:first_start].decode("utf-8")
        fragment_suffix = source_bytes[last_end:].decode("utf-8") + suffix_context

        start_index = 0
        while start_index < len(children):
            end_index = start_index
            while end_index < len(children):
                segment_start = (
                    child_spans[start_index - 1][1]
                    if start_index > 0
                    else child_spans[0][0]
                )
                segment_end = child_spans[end_index][1]
                code = (
                    fragment_prefix
                    + source_bytes[segment_start:segment_end].decode("utf-8")
                    + fragment_suffix
                )
                if _candidate_fits(code, artifact, feature_type, model, budget, encoding):
                    end_index += 1
                    continue

                if end_index > start_index:
                    accepted_end = child_spans[end_index - 1][1]
                    accepted_start = (
                        child_spans[start_index - 1][1]
                        if start_index > 0
                        else child_spans[0][0]
                    )
                    accepted_code = (
                        fragment_prefix
                        + source_bytes[accepted_start:accepted_end].decode("utf-8")
                        + fragment_suffix
                    )
                    fragments.append(
                        (
                            accepted_code,
                            base_offset + accepted_start,
                            base_offset + accepted_end,
                            fragment_prefix,
                        )
                    )
                    start_index = end_index
                    break

                child_start, child_end = child_spans[start_index]
                leading_gap_start = (
                    child_spans[start_index - 1][1]
                    if start_index > 0
                    else child_start
                )
                nested_prefix = fragment_prefix + source_bytes[
                    leading_gap_start:child_start
                ].decode("utf-8")
                nested_suffix = source_bytes[child_end:segment_end].decode("utf-8") + fragment_suffix
                child_text = source_bytes[child_start:child_end].decode("utf-8")
                before_split_count = len(fragments)
                split_text(
                    child_text,
                    nested_prefix,
                    nested_suffix,
                    base_offset + child_start,
                    depth + 1,
                )
                if len(fragments) == before_split_count:
                    raise SourceSplitError(
                        f"Indivisible syntax node exceeds the prompt budget in {artifact.uri}"
                    )
                start_index += 1
                break
            else:
                accepted_end = child_spans[end_index - 1][1]
                accepted_start = (
                    child_spans[start_index - 1][1]
                    if start_index > 0
                    else child_spans[0][0]
                )
                accepted_code = (
                    fragment_prefix
                    + source_bytes[accepted_start:accepted_end].decode("utf-8")
                    + fragment_suffix
                )
                fragments.append(
                    (
                        accepted_code,
                        base_offset + accepted_start,
                        base_offset + accepted_end,
                        fragment_prefix,
                    )
                )
                start_index = end_index

    split_text(artifact.code, "", "", 0, 0)
    chunks = [
        SourceChunk(
            original_artifact_uri=artifact.uri,
            file_uri=artifact.file_uri,
            language=artifact.language,
            part_index=index,
            start_byte=start_byte,
            end_byte=end_byte,
            context=context,
            code=code,
        )
        for index, (code, start_byte, end_byte, context) in enumerate(fragments, start=1)
    ]
    for chunk in chunks:
        check_feature_prompt_budget(
            render_feature_messages({chunk.prompt_key: chunk.code}, feature_type),
            model,
            "source-split",
            budget=budget,
            encoding=encoding,
        )
    return chunks


def pack_source_artifacts(
    artifacts,
    feature_type,
    model,
    budget=None,
    encoding=None,
):
    """Pack complete source artifacts into deterministic, budgeted prompt payloads."""
    budget = budget or get_feature_prompt_budget()
    ordered_artifacts = sorted(artifacts, key=lambda item: (item.file_uri, item.uri))
    batches = []
    current_batch = []

    def flush_batch():
        if current_batch:
            batches.append(SourceBatch(tuple(current_batch)))
            current_batch.clear()

    def append_chunk(chunk):
        candidate = current_batch + [chunk]
        candidate_payload = {item.prompt_key: item.code for item in candidate}
        try:
            check_feature_prompt_budget(
                render_feature_messages(candidate_payload, feature_type),
                model,
                "source-batch",
                budget=budget,
                encoding=encoding,
            )
        except FeaturePromptTooLarge:
            flush_batch()
            check_feature_prompt_budget(
                render_feature_messages({chunk.prompt_key: chunk.code}, feature_type),
                model,
                "source-batch",
                budget=budget,
                encoding=encoding,
            )
            current_batch.append(chunk)
        else:
            current_batch.append(chunk)

    for artifact in ordered_artifacts:
        chunk = SourceChunk(
            original_artifact_uri=artifact.uri,
            file_uri=artifact.file_uri,
            language=artifact.language,
            part_index=0,
            start_byte=0,
            end_byte=len(artifact.code.encode("utf-8")),
            context="",
            code=artifact.code,
        )
        try:
            check_feature_prompt_budget(
                render_feature_messages(
                    {item.prompt_key: item.code for item in current_batch + [chunk]},
                    feature_type,
                ),
                model,
                "source-batch",
                budget=budget,
                encoding=encoding,
            )
        except FeaturePromptTooLarge:
            flush_batch()
            try:
                check_feature_prompt_budget(
                    render_feature_messages({chunk.prompt_key: chunk.code}, feature_type),
                    model,
                    "source-batch",
                    budget=budget,
                    encoding=encoding,
                )
            except FeaturePromptTooLarge:
                for split_chunk in split_oversized_artifact(
                    artifact, feature_type, model, budget=budget, encoding=encoding
                ):
                    append_chunk(split_chunk)
            else:
                append_chunk(chunk)
        else:
            current_batch.append(chunk)

    flush_batch()
    return batches