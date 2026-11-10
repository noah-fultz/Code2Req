from django.test import SimpleTestCase
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from c2c_web.utils.process.process_repo.feature_generation import (
    FeaturePromptBudget,
    FeaturePromptTooLarge,
    estimate_feature_message_tokens,
    render_feature_messages,
)
from c2c_web.utils.process.process_repo.source_chunks import (
    ArtifactCollectionError,
    SourceArtifact,
    SourceSplitError,
    collect_cluster_artifacts,
    pack_source_artifacts,
)


class CharacterEncoding:
    def encode(self, text):
        return list(text)


class SourceArtifactTests(SimpleTestCase):
    def setUp(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/repo#")
        self.file_a = URIRef("http://example.org/repo/src/a.c")
        self.file_b = URIRef("http://example.org/repo/src/b.py")
        self.artifact_a = URIRef(f"{self.file_a}#first")
        self.artifact_b = URIRef(f"{self.file_a}#second")
        self.artifact_c = URIRef(f"{self.file_b}#third")
        self._add_artifact(self.file_a, self.artifact_a, "Function", "first", "int first() {}")
        self._add_artifact(self.file_a, self.artifact_b, "Function", "second", "int second() {}")
        self._add_artifact(self.file_b, self.artifact_c, "Snippet", "third", "x = 1")

    def _add_artifact(self, file_uri, artifact_uri, kind, name, code):
        self.graph.add((file_uri, RDF.type, self.ex.CodeFile))
        self.graph.add((file_uri, self.ex.containsArtifact, artifact_uri))
        self.graph.add((artifact_uri, RDF.type, getattr(self.ex, kind)))
        self.graph.add((artifact_uri, self.ex.name, Literal(name)))
        self.graph.add((artifact_uri, self.ex.code, Literal(code)))

    def test_collection_resolves_owner_and_metadata_from_graph(self):
        records = collect_cluster_artifacts(
            self.graph,
            self.ex,
            [str(self.artifact_c), str(self.artifact_a), str(self.artifact_b)],
        )

        self.assertEqual([record.uri for record in records], [
            str(self.artifact_a), str(self.artifact_b), str(self.artifact_c)
        ])
        self.assertEqual(records[0].file_uri, str(self.file_a))
        self.assertEqual(records[0].language, "c")
        self.assertEqual(records[0].kind, "Function")
        self.assertEqual(records[0].name, "first")
        self.assertEqual(records[0].code, "int first() {}")
        self.assertEqual(records[2].language, "py")

    def test_missing_or_ambiguous_owner_is_rejected(self):
        orphan = URIRef("http://example.org/repo/orphan.c#work")
        with self.assertRaises(ArtifactCollectionError):
            collect_cluster_artifacts(self.graph, self.ex, [str(orphan)])

        other_file = URIRef("http://example.org/repo/src/other.c")
        self.graph.add((other_file, RDF.type, self.ex.CodeFile))
        self.graph.add((other_file, self.ex.containsArtifact, self.artifact_a))
        with self.assertRaises(ArtifactCollectionError):
            collect_cluster_artifacts(self.graph, self.ex, [str(self.artifact_a)])

    def test_missing_or_multiple_code_values_are_rejected(self):
        no_code = URIRef("http://example.org/repo/src/a.c#no_code")
        self.graph.add((self.file_a, self.ex.containsArtifact, no_code))
        self.graph.add((no_code, RDF.type, self.ex.Function))
        with self.assertRaises(ArtifactCollectionError):
            collect_cluster_artifacts(self.graph, self.ex, [str(no_code)])

        self.graph.add((self.artifact_a, self.ex.code, Literal("different code")))
        with self.assertRaises(ArtifactCollectionError):
            collect_cluster_artifacts(self.graph, self.ex, [str(self.artifact_a)])

    def test_packing_is_deterministic_and_every_batch_fits(self):
        records = collect_cluster_artifacts(
            self.graph,
            self.ex,
            [str(self.artifact_c), str(self.artifact_b), str(self.artifact_a)],
        )
        encoding = CharacterEncoding()
        first_two = {record.uri: record.code for record in records[:2]}
        max_input_tokens = estimate_feature_message_tokens(
            render_feature_messages(first_two, "Code"), encoding
        )
        budget = FeaturePromptBudget(
            input_token_limit=max_input_tokens,
            context_window=max_input_tokens + 100,
            output_reserve=50,
            safety_margin=50,
        )

        batches = pack_source_artifacts(
            list(reversed(records)), "Code", "qwen-test", budget, encoding
        )

        self.assertEqual([list(batch.payload) for batch in batches], [
            [str(self.artifact_a), str(self.artifact_b)],
            [str(self.artifact_c)],
        ])
        self.assertEqual(
            sorted(uri for batch in batches for uri in batch.payload),
            sorted(record.uri for record in records),
        )
        for batch in batches:
            count = estimate_feature_message_tokens(
                render_feature_messages(batch.payload, "Code"), encoding
            )
            self.assertLessEqual(count, budget.max_input_tokens)

    def test_oversized_artifacts_split_at_syntax_boundaries(self):
        encoding = CharacterEncoding()
        examples = [
            (
                "c",
                "Function",
                "int calculate(int value) {\n"
                "    int first = value + 1;\n"
                "    int second = first + 2;\n"
                "    int third = second + 3;\n"
                "    int fourth = third + 4;\n"
                "    return fourth;\n"
                "}\n",
            ),
            (
                "py",
                "Function",
                "def calculate(value):\n"
                "    first = value + 1\n"
                "    second = first + 2\n"
                "    third = second + 3\n"
                "    fourth = third + 4\n"
                "    return fourth\n",
            ),
            (
                "java",
                "Function",
                "int calculate(int value) {\n"
                "    int first = value + 1;\n"
                "    int second = first + 2;\n"
                "    int third = second + 3;\n"
                "    int fourth = third + 4;\n"
                "    return fourth;\n"
                "}\n",
            ),
        ]

        for language, kind, code in examples:
            with self.subTest(language=language):
                uri = f"http://example.org/repo/src/calculate.{language}#calculate"
                record = SourceArtifact(
                    uri=uri,
                    file_uri=f"http://example.org/repo/src/calculate.{language}",
                    language=language,
                    kind=kind,
                    name="calculate",
                    code=code,
                )
                empty_count = estimate_feature_message_tokens(
                    render_feature_messages({f"{uri}::part-00000001": ""}, "Code"),
                    encoding,
                )
                limit = empty_count + 80
                budget = FeaturePromptBudget(
                    input_token_limit=limit,
                    context_window=limit + 100,
                    output_reserve=50,
                    safety_margin=50,
                )

                batches = pack_source_artifacts(
                    [record], "Code", "qwen-test", budget, encoding
                )
                chunks = [chunk for batch in batches for chunk in batch.chunks]

                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(chunk.original_artifact_uri == uri for chunk in chunks))
                self.assertEqual([chunk.part_index for chunk in chunks], list(range(1, len(chunks) + 1)))
                for batch in batches:
                    count = estimate_feature_message_tokens(
                        render_feature_messages(batch.payload, "Code"), encoding
                    )
                    self.assertLessEqual(count, budget.max_input_tokens)
                for statement in ("first", "second", "third", "fourth", "return fourth"):
                    self.assertTrue(any(statement in chunk.code for chunk in chunks), statement)
                self.assertTrue(all(chunk.start_byte < chunk.end_byte for chunk in chunks))

    def test_indivisible_or_unparseable_oversized_artifact_fails_explicitly(self):
        record = SourceArtifact(
            uri="http://example.org/repo/src/large.c#large",
            file_uri="http://example.org/repo/src/large.c",
            language="c",
            kind="Function",
            name="large",
            code="int large() { return 1; }",
        )
        budget = FeaturePromptBudget(
            input_token_limit=1,
            context_window=100,
            output_reserve=10,
            safety_margin=10,
        )

        with self.assertRaises(SourceSplitError):
            pack_source_artifacts([record], "Code", "qwen-test", budget, CharacterEncoding())

    def test_oversized_nested_block_is_split_recursively(self):
        code = (
            "int nested(int value) {\n"
            "    if (value > 0) {\n"
            "        int first = value + 1;\n"
            "        int second = first + 2;\n"
            "        int third = second + 3;\n"
            "        int fourth = third + 4;\n"
            "        int fifth = fourth + 5;\n"
            "    }\n"
            "    return value;\n"
            "}\n"
        )
        record = SourceArtifact(
            uri="http://example.org/repo/src/nested.c#nested",
            file_uri="http://example.org/repo/src/nested.c",
            language="c",
            kind="Function",
            name="nested",
            code=code,
        )
        encoding = CharacterEncoding()
        empty_count = estimate_feature_message_tokens(
            render_feature_messages({f"{record.uri}::part-00000001": ""}, "Code"),
            encoding,
        )
        limit = empty_count + 130
        budget = FeaturePromptBudget(
            input_token_limit=limit,
            context_window=limit + 100,
            output_reserve=50,
            safety_margin=50,
        )

        batches = pack_source_artifacts([record], "Code", "qwen-test", budget, encoding)
        chunks = [chunk for batch in batches for chunk in batch.chunks]

        self.assertGreater(len(chunks), 1)
        for statement in ("first", "second", "third", "fourth", "fifth"):
            self.assertTrue(any(f"int {statement} =" in chunk.code for chunk in chunks))
        self.assertTrue(all(chunk.original_artifact_uri == record.uri for chunk in chunks))