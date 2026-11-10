from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from c2c_web.utils.prompts import Feature
from c2c_web.utils.create_and_analyze import analyze as analyze_module
from c2c_web.utils.process.process_repo import cluster_code, join_features
from c2c_web.utils.process.process_repo.feature_generation import (
    FeaturePromptBudget,
    FeatureReductionError,
    estimate_feature_message_tokens,
    pack_feature_records,
    render_feature_messages,
    summarize_bounded,
)
from c2c_web.utils.process.process_repo.source_chunks import SourceBatch, SourceChunk


class CharacterEncoding:
    def encode(self, text):
        return list(text)


class BoundedReductionTests(SimpleTestCase):
    def test_multiple_code_batches_reduce_to_one_feature(self):
        encoding = CharacterEncoding()
        initial_batches = [
            {f"source-{index}": f"small code {index}"}
            for index in range(3)
        ]
        mapped_summaries = {
            f"summary_{index:08d}": f"Map {index}: " + ("x" * 120)
            for index in range(3)
        }
        pair_count = estimate_feature_message_tokens(
            render_feature_messages(dict(list(mapped_summaries.items())[:2]), "Features"),
            encoding,
        )
        map_count = max(
            estimate_feature_message_tokens(render_feature_messages(batch, "Code"), encoding)
            for batch in initial_batches
        )
        limit = max(pair_count, map_count)
        budget = FeaturePromptBudget(limit, limit + 100, 50, 50)
        calls = []
        map_index = 0

        def summarize_once(payload, mode, model, stage):
            nonlocal map_index
            calls.append((mode, payload))
            if mode == "Code":
                feature = Feature(name=f"Map {map_index}", description="x" * 120)
                map_index += 1
            else:
                feature = Feature(name="Reduced", description="short combined summary")
            return feature, 0.75

        feature, score = summarize_bounded(
            {},
            "Code",
            "qwen-test",
            "cluster-test",
            summarize_once,
            budget=budget,
            encoding=encoding,
            initial_batches=initial_batches,
        )

        self.assertEqual(feature.name, "Reduced")
        self.assertEqual(score, 0.75)
        self.assertEqual([mode for mode, _ in calls], ["Code", "Code", "Code", "Features", "Features", "Features"])
        self.assertTrue(all(len(payload) > 0 for _, payload in calls))

    def test_reduction_fails_when_batches_cannot_be_combined(self):
        encoding = CharacterEncoding()
        initial_batches = [{"source-a": "code a"}, {"source-b": "code b"}]
        one_summary = {"summary_00000000": "Mapped: " + ("x" * 150)}
        single_count = estimate_feature_message_tokens(
            render_feature_messages(one_summary, "Features"), encoding
        )
        map_count = max(
            estimate_feature_message_tokens(render_feature_messages(batch, "Code"), encoding)
            for batch in initial_batches
        )
        budget = FeaturePromptBudget(max(single_count, map_count), 10000, 50, 50)
        calls = []

        def summarize_once(payload, mode, model, stage):
            calls.append(mode)
            return Feature(name="Mapped", description="x" * 150), 0.5

        with self.assertRaises(FeatureReductionError):
            summarize_bounded(
                {},
                "Code",
                "qwen-test",
                "cluster-test",
                summarize_once,
                budget=budget,
                encoding=encoding,
                initial_batches=initial_batches,
            )

        self.assertEqual(calls, ["Code", "Code"])

    def test_summary_record_packer_rejects_an_oversized_single_record(self):
        budget = FeaturePromptBudget(1, 100, 10, 10)

        with self.assertRaises(ValueError):
            pack_feature_records({"summary": "too large"}, "Features", "qwen-test", budget)


class RepositoryFeatureGraphTests(SimpleTestCase):
    def setUp(self):
        self.graph = Graph()
        self.ex = Namespace("http://example.org/test#")

    def add_code_artifact(self, file_uri, artifact_uri, name, code):
        self.graph.add((file_uri, RDF.type, self.ex.CodeFile))
        self.graph.add((file_uri, self.ex.containsArtifact, artifact_uri))
        self.graph.add((artifact_uri, RDF.type, self.ex.Function))
        self.graph.add((artifact_uri, self.ex.name, Literal(name)))
        self.graph.add((artifact_uri, self.ex.code, Literal(code)))

    def test_cluster_map_reduce_links_one_feature_to_original_artifacts(self):
        file_uri = URIRef("http://example.org/repo/src/example.c")
        artifact_uris = [
            URIRef(f"{file_uri}#first"),
            URIRef(f"{file_uri}#second"),
        ]
        for name, artifact_uri in zip(("first", "second"), artifact_uris):
            self.add_code_artifact(file_uri, artifact_uri, name, f"int {name}() {{ return 1; }}")

        batches = [
            SourceBatch((
                SourceChunk(str(uri), str(file_uri), "c", 0, 0, 10, "", f"code {index}"),
            ))
            for index, uri in enumerate(artifact_uris)
        ]
        generated = [
            (Feature(name="First batch", description="first summary"), 0.8),
            (Feature(name="Second batch", description="second summary"), 0.8),
            (Feature(name="Combined", description="combined behavior"), 0.8),
        ]

        with (
            patch.object(cluster_code, "get_embedded", return_value={}),
            patch.object(cluster_code, "get_artifact_edgelist", return_value=[]),
            patch.object(cluster_code, "pack_source_artifacts", return_value=batches),
            patch.object(cluster_code, "get_description", side_effect=generated) as describe,
        ):
            cluster_code.cluster_code_artifacts(
                self.graph,
                "http://example.org/test",
                "/tmp/repo",
                "qwen-test",
            )

        generated_features = list(self.graph.subjects(RDF.type, self.ex.Feature))
        self.assertEqual(len(generated_features), 1)
        feature_uri = generated_features[0]
        self.assertEqual(str(self.graph.value(feature_uri, self.ex.level)), "2")
        self.assertEqual(
            set(self.graph.objects(feature_uri, self.ex.isFeatureFor)),
            set(artifact_uris),
        )
        self.assertEqual([call.args[1] for call in describe.call_args_list], [
            "Code", "Code", "Features"
        ])

    def _add_feature(self, uri, name, description):
        self.graph.add((uri, RDF.type, self.ex.Feature))
        self.graph.add((uri, self.ex.name, Literal(name)))
        self.graph.add((uri, self.ex.description, Literal(description)))
        self.graph.add((uri, self.ex.level, Literal("2")))

    def test_join_group_and_root_keep_expected_link_levels(self):
        feature_uris = [
            URIRef("http://example.org/repo#feature_a"),
            URIRef("http://example.org/repo#feature_b"),
        ]
        for index, feature_uri in enumerate(feature_uris):
            artifact_uri = URIRef(f"http://example.org/repo/src/a.c#artifact{index}")
            self._add_feature(feature_uri, f"Feature {index}", f"Description {index}")
            self.graph.add((feature_uri, self.ex.isFeatureFor, artifact_uri))

        generated = [
            (Feature(name="Joined", description="joined descriptions"), 0.9),
            (Feature(name="Root", description="root description"), 0.9),
        ]
        feature_map = {str(uri): [float(index), 1.0] for index, uri in enumerate(feature_uris)}
        paths = {
            str(feature_uris[0]): {str(feature_uris[0]): 0, str(feature_uris[1]): 1},
            str(feature_uris[1]): {str(feature_uris[0]): 1, str(feature_uris[1]): 0},
        }

        with (
            patch.object(join_features, "get_feature_embeddings", return_value=feature_map),
            patch.object(join_features, "feature_shortest_paths", return_value=paths),
            patch.object(join_features, "compute_similarity_matrix"),
            patch.object(
                join_features,
                "similarity_to_infomap_edges",
                return_value=([(0, 1, 1.0)], [str(uri) for uri in feature_uris]),
            ),
            patch.object(join_features, "run_infomap_on_similarity", return_value={1: [str(uri) for uri in feature_uris]}),
            patch.object(join_features, "get_description", side_effect=generated) as describe,
        ):
            join_features.join_features(self.graph, "http://example.org/test", "/tmp/repo", "qwen-test")

        level_one = next(
            uri for uri in self.graph.subjects(RDF.type, self.ex.Feature)
            if str(self.graph.value(uri, self.ex.level)) == "1"
        )
        level_zero = next(
            uri for uri in self.graph.subjects(RDF.type, self.ex.Feature)
            if str(self.graph.value(uri, self.ex.level)) == "0"
        )
        self.assertEqual(set(self.graph.objects(level_one, self.ex.isFeatureFor)), set(feature_uris))
        self.assertEqual(set(self.graph.objects(level_zero, self.ex.isFeatureFor)), {level_one})
        self.assertEqual([call.args[1] for call in describe.call_args_list], ["Features", "Features"])

    def test_empty_and_single_feature_inputs_skip_join_model_calls(self):
        with patch.object(join_features, "get_feature_embeddings") as embeddings:
            self.assertIsNone(
                join_features.join_features(self.graph, "http://example.org/test", "/tmp/repo", "qwen-test")
            )
            embeddings.assert_not_called()

            single_uri = URIRef("http://example.org/repo#only_feature")
            self._add_feature(single_uri, "Only", "One feature")
            result = join_features.join_features(
                self.graph, "http://example.org/test", "/tmp/repo", "qwen-test"
            )

        self.assertEqual(result, str(single_uri))
        embeddings.assert_not_called()

    def test_disconnected_features_pass_through_to_bounded_root(self):
        feature_uris = [
            URIRef("http://example.org/repo#disconnected_a"),
            URIRef("http://example.org/repo#disconnected_b"),
        ]
        for index, feature_uri in enumerate(feature_uris):
            self._add_feature(feature_uri, f"Feature {index}", f"Description {index}")
        with (
            patch.object(join_features, "get_feature_embeddings", return_value={str(uri): [1.0] for uri in feature_uris}),
            patch.object(join_features, "feature_shortest_paths", return_value={}),
            patch.object(join_features, "compute_similarity_matrix"),
            patch.object(
                join_features,
                "similarity_to_infomap_edges",
                return_value=([], [str(uri) for uri in feature_uris]),
            ),
            patch.object(join_features, "get_description", return_value=(Feature(name="Root", description="All features"), 0.9)) as describe,
        ):
            join_features.join_features(self.graph, "http://example.org/test", "/tmp/repo", "qwen-test")

        roots = [
            uri for uri in self.graph.subjects(RDF.type, self.ex.Feature)
            if str(self.graph.value(uri, self.ex.level)) == "0"
        ]
        self.assertEqual(len(roots), 1)
        self.assertEqual(set(self.graph.objects(roots[0], self.ex.isFeatureFor)), set(feature_uris))
        self.assertEqual(describe.call_count, 1)

    def test_repeated_infomap_partition_retries_then_terminates(self):
        members = ["artifact-a", "artifact-b", "artifact-c", "artifact-d"]
        partitions = [
            {1: members},
            {2: members},
        ]
        final_cluster_map = {}
        edge_list = [(members[0], members[1], 1), (members[2], members[3], 1)]

        with (
            patch.object(cluster_code, "run_infomap", side_effect=partitions) as run_infomap,
            patch.object(cluster_code, "get_artifact_edgelist", return_value=edge_list),
        ):
            cluster_ids = cluster_code.recursive_cluster_maker(
                final_cluster_map,
                edge_list,
                {},
                preferred_cluster_size=2,
                graph=self.graph,
                EX=self.ex,
                max_depth=5,
                depth=0,
                prev_clusters={0: members},
            )

        self.assertEqual(len(cluster_ids), 1)
        self.assertEqual(final_cluster_map[cluster_ids[0]]["members"], members)
        self.assertEqual(run_infomap.call_count, 2)
        self.assertTrue(run_infomap.call_args_list[1].args[2])

    def test_repository_orchestration_passes_one_budget_to_both_stages(self):
        project = SimpleNamespace(name="budget-project", clusters=4, model="qwen2.5-coder:32b")
        feature_budget = FeaturePromptBudget(1000, 2000, 300, 100)
        fake_graph = SimpleNamespace(serialize=lambda *args, **kwargs: None, parse=lambda *args, **kwargs: None)

        with (
            patch.object(analyze_module, "get_vector_collection_names", return_value=("code", "feature", "req")),
            patch.object(analyze_module, "get_feature_prompt_budget", return_value=feature_budget),
            patch.object(analyze_module, "build_code_repository_graph", return_value=(fake_graph, 10)),
            patch.object(analyze_module, "Graph", return_value=fake_graph),
            patch.object(analyze_module, "cluster_code_artifacts") as cluster_call,
            patch.object(analyze_module, "join_features") as join_call,
            patch.object(analyze_module, "upload_ttl_to_graphdb"),
            patch.object(analyze_module, "embed_features_desc_code"),
            patch.object(analyze_module, "get_stats"),
        ):
            analyze_module.start_repository_analysis(project)

        self.assertIs(cluster_call.call_args.kwargs["feature_budget"], feature_budget)
        self.assertIs(join_call.call_args.kwargs["feature_budget"], feature_budget)
