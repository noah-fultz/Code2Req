import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from c2c_web.models import EvaluationFile, EvaluationUnit, Project
from c2c_web.utils.alignment.alignment import get_alignment
from c2c_web.utils.create_and_analyze import analyze as analyze_module
from c2c_web.utils.process.process_req import compare as compare_module
from c2c_web.utils.process.process_req import process_eval_file as process_module


class AlignmentTests(TestCase):
    def test_alignment_preserves_composite_links_and_handles_unlinked_units(self):
        project = Project.objects.create(
            name="alignment-project",
            github_repo="https://example.com/alignment-project",
        )
        reference_file = EvaluationFile.objects.create(
            project=project,
            name="Reference requirements",
            path="",
        )
        generated_file = EvaluationFile.objects.create(
            project=project,
            name="Code_Features",
            path="",
        )
        linked_unit = EvaluationUnit.objects.create(
            file=generated_file,
            identifier="feature_req_1",
            description="A generated requirement.",
            linked_feature_ids=[
                f"{reference_file.id}:REQ-1",
                "Code_Features_RAG:feature_RAG_req_1",
            ],
            linked_code=["http://example.org/code#search"],
            scores=["0.95", "0.91"],
        )
        unlinked_unit = EvaluationUnit.objects.create(
            file=generated_file,
            identifier="feature_req_2",
            description="Another generated requirement.",
        )

        result = get_alignment(project)

        generated_alignment = next(section for section in result if section["name"] == "From Feature")
        rows = generated_alignment["alignments"]
        linked_row = next(row for row in rows if row["source"] == linked_unit.identifier)
        unlinked_row = next(row for row in rows if row["source"] == unlinked_unit.identifier)
        self.assertEqual(
            linked_row["linked_sources"],
            [f"{reference_file.id}:REQ-1", "Code_Features_RAG:feature_RAG_req_1"],
        )
        self.assertEqual(linked_row["linked_code"], ["http://example.org/code#search"])
        self.assertEqual(linked_row["scores"], ["0.95", "0.91"])
        self.assertEqual(linked_row["linked"], "Yes")
        self.assertEqual(unlinked_row["linked_sources"], [])
        self.assertEqual(unlinked_row["linked"], "No")


class RequirementComparisonTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="test-project",
            github_repo="https://example.com/test-project",
            use_feature=True,
            use_code=False,
            use_rag=False,
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_dir = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def add_reference(self, name="Example requirements", identifier="REQ-1"):
        file_record = EvaluationFile.objects.create(
            project=self.project,
            name=name,
            path="",
        )
        unit = EvaluationUnit.objects.create(
            file=file_record,
            identifier=identifier,
            description="The clinic staff can search for an owner.",
        )
        return file_record, unit

    def add_generated(self, name="Code_Features", identifier="feature_req_1"):
        file_record = EvaluationFile.objects.create(
            project=self.project,
            name=name,
            path="",
        )
        unit = EvaluationUnit.objects.create(
            file=file_record,
            identifier=identifier,
            description="Clinic staff can search for an owner.",
        )
        return file_record, unit

    @patch("c2c_web.utils.process.process_req.compare.cosine_similarity", return_value=0.95)
    @patch("c2c_web.utils.process.process_req.compare.get_embeddings")
    def test_feature_only_compares_and_reports_only_feature_variant(self, mock_embeddings, _mock_similarity):
        self.add_reference()
        generated_file, _ = self.add_generated()
        mock_embeddings.side_effect = lambda descriptions: {
            identifier: [1.0, 0.0] for identifier in descriptions
        }

        compare_module.compare(self.project, str(self.report_dir))

        self.assertEqual(generated_file.refresh_from_db(), None)
        self.assertEqual(generated_file.f1_score, 1.0)
        self.assertTrue((self.report_dir / "generated_reqs_features.tsv").exists())
        self.assertFalse((self.report_dir / "generated_reqs_code.tsv").exists())
        self.assertFalse((self.report_dir / "generated_reqs_features_rag.tsv").exists())
        self.assertEqual(len(list(self.report_dir.glob("reference_*_linked.tsv"))), 1)

    def test_missing_enabled_variant_fails_with_actionable_error(self):
        self.add_reference()

        with self.assertRaisesRegex(ValueError, "Code_Features.*did not create"):
            compare_module.compare(self.project, str(self.report_dir))

    @patch("c2c_web.utils.process.process_req.compare.get_embeddings", return_value={})
    def test_disabled_stale_variant_is_ignored(self, _mock_embeddings):
        self.add_reference()
        self.project.use_feature = False
        self.project.use_code = False
        self.project.save()
        self.add_generated(name="Req_from_code")

        compare_module.compare(self.project, str(self.report_dir))

        self.assertFalse((self.report_dir / "generated_reqs_code.tsv").exists())
        self.assertEqual(list(self.report_dir.glob("reference_*_linked.tsv"))[0].read_text().splitlines()[0], "#id\tRequirement\tlinked")

    @patch("c2c_web.utils.process.process_req.compare.cosine_similarity", return_value=0.95)
    @patch("c2c_web.utils.process.process_req.compare.get_embeddings")
    def test_duplicate_reference_ids_are_scoped_by_file(self, mock_embeddings, _mock_similarity):
        _, first_unit = self.add_reference(name="First", identifier="REQ-1")
        _, second_unit = self.add_reference(name="Second", identifier="REQ-1")
        _, generated_unit = self.add_generated()
        mock_embeddings.side_effect = lambda descriptions: {
            identifier: [1.0, 0.0] for identifier in descriptions
        }

        compare_module.compare(self.project, str(self.report_dir))

        first_unit.refresh_from_db()
        second_unit.refresh_from_db()
        generated_unit.refresh_from_db()
        self.assertEqual(len(first_unit.linked_feature_ids), 1)
        self.assertEqual(len(second_unit.linked_feature_ids), 1)
        self.assertEqual(len(generated_unit.linked_feature_ids), 2)
        self.assertNotEqual(generated_unit.linked_feature_ids[0], generated_unit.linked_feature_ids[1])
        self.assertEqual(len(list(self.report_dir.glob("reference_*_linked.tsv"))), 2)

    @patch("c2c_web.utils.process.process_req.compare.cosine_similarity", return_value=0.0)
    @patch("c2c_web.utils.process.process_req.compare.get_embeddings")
    def test_empty_enabled_variant_is_valid_and_scores_zero(self, mock_embeddings, _mock_similarity):
        reference_file, _ = self.add_reference()
        generated_file = EvaluationFile.objects.create(
            project=self.project,
            name="Code_Features",
            path="",
        )
        mock_embeddings.return_value = {}

        compare_module.compare(self.project, str(self.report_dir))

        generated_file.refresh_from_db()
        reference_file.refresh_from_db()
        self.assertEqual(generated_file.f1_score, 0.0)
        self.assertEqual(generated_file.num_units, 0)
        self.assertEqual(reference_file.f1_score, 0.0)
        self.assertEqual(reference_file.num_linked_to_feature, 0)
        self.assertTrue((self.report_dir / "generated_reqs_features.tsv").exists())


class RequirementGenerationTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="test-project",
            github_repo="https://example.com/test-project",
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    @patch("c2c_web.utils.process.process_req.process_eval_file.get_reqs_from_feature")
    def test_feature_generation_runs_without_reference_files(self, mock_generate):
        mock_generate.return_value = ([SimpleNamespace(description="A generated requirement.")], [])

        has_references = process_module.process_eval_files(
            self.project,
            graph=None,
            ex="http://example.org/code-repository",
            repo_root=self.temp_dir.name,
            vector_collection_requirements="requirements-test",
            vector_path=self.temp_dir.name,
            model="test-model",
            top_k=2,
            max_tokens=1000,
            rag=True,
            use_feature=True,
            use_code=False,
        )

        generated = EvaluationFile.objects.get(project=self.project, name="Code_Features")
        self.assertFalse(has_references)
        self.assertEqual(generated.units.count(), 1)
        self.assertEqual(generated.units.get().description, "A generated requirement.")
        self.assertFalse(EvaluationFile.objects.filter(project=self.project, name="Code_Features_RAG").exists())
        mock_generate.assert_called_once()

    @patch("c2c_web.utils.process.process_req.process_eval_file.add_to_vectordb")
    @patch("c2c_web.utils.process.process_req.process_eval_file.get_or_create_collection")
    @patch("c2c_web.utils.process.process_req.process_eval_file.delete_collection")
    @patch("c2c_web.utils.process.process_req.process_eval_file.chromadb.PersistentClient")
    @patch("c2c_web.utils.process.process_req.process_eval_file.get_reqs_from_feature")
    def test_reference_csv_and_feature_variant_are_processed_once(
        self,
        mock_generate,
        _mock_client,
        mock_delete_collection,
        mock_get_collection,
        mock_add,
    ):
        csv_path = Path(self.temp_dir.name) / "requirements.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("ID", "Description"))
            writer.writerow(("REQ-1", "The system can search for an owner."))
        EvaluationFile.objects.create(
            project=self.project,
            name="Example requirements",
            path=str(csv_path),
        )
        mock_get_collection.return_value = object()
        mock_generate.return_value = ([SimpleNamespace(description="A generated requirement.")], [])

        has_references = process_module.process_eval_files(
            self.project,
            graph=None,
            ex="http://example.org/code-repository",
            repo_root=self.temp_dir.name,
            vector_collection_requirements="requirements-test",
            vector_path=self.temp_dir.name,
            model="test-model",
            top_k=2,
            max_tokens=1000,
            rag=False,
            use_feature=True,
            use_code=False,
        )

        self.assertTrue(has_references)
        self.assertEqual(mock_delete_collection.call_count, 1)
        self.assertEqual(mock_add.call_count, 1)
        mock_generate.assert_called_once()
        reference_file = EvaluationFile.objects.get(name="Example requirements")
        self.assertEqual(reference_file.units.count(), 1)
        self.assertTrue(EvaluationFile.objects.filter(project=self.project, name="Code_Features").exists())

    @patch("c2c_web.utils.process.process_req.process_eval_file.add_to_vectordb")
    @patch("c2c_web.utils.process.process_req.process_eval_file.get_or_create_collection")
    @patch("c2c_web.utils.process.process_req.process_eval_file.delete_collection")
    @patch("c2c_web.utils.process.process_req.process_eval_file.chromadb.PersistentClient")
    @patch("c2c_web.utils.process.process_req.process_eval_file.get_reqs_from_feature")
    def test_rag_creates_only_enabled_feature_variants(
        self,
        mock_generate,
        _mock_client,
        _mock_delete_collection,
        mock_get_collection,
        _mock_add,
    ):
        csv_path = Path(self.temp_dir.name) / "requirements.csv"
        csv_path.write_text("ID,Description\nREQ-1,An owner can be searched.\n", encoding="utf-8")
        EvaluationFile.objects.create(
            project=self.project,
            name="Example requirements",
            path=str(csv_path),
        )
        mock_get_collection.return_value = object()
        mock_generate.return_value = ([SimpleNamespace(description="A generated requirement.")], [])

        process_module.process_eval_files(
            self.project,
            graph=None,
            ex="http://example.org/code-repository",
            repo_root=self.temp_dir.name,
            vector_collection_requirements="requirements-test",
            vector_path=self.temp_dir.name,
            model="test-model",
            top_k=2,
            max_tokens=1000,
            rag=True,
            use_feature=True,
            use_code=False,
        )

        self.assertEqual(mock_generate.call_count, 2)
        self.assertFalse(mock_generate.call_args_list[0].args[10])
        self.assertTrue(mock_generate.call_args_list[1].args[10])
        self.assertTrue(EvaluationFile.objects.filter(project=self.project, name="Code_Features").exists())
        self.assertTrue(EvaluationFile.objects.filter(project=self.project, name="Code_Features_RAG").exists())
        self.assertFalse(EvaluationFile.objects.filter(project=self.project, name="Req_from_code").exists())
        self.assertFalse(EvaluationFile.objects.filter(project=self.project, name="Req_from_code_RAG").exists())


class RequirementAnalysisOrchestrationTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="test-project",
            github_repo="https://example.com/test-project",
        )

    @patch("c2c_web.utils.create_and_analyze.analyze.compare")
    @patch("c2c_web.utils.create_and_analyze.analyze.process_eval_files", return_value=False)
    @patch("c2c_web.utils.create_and_analyze.analyze.get_vector_collection_names", return_value=("code", "feature", "requirements"))
    @patch("c2c_web.utils.create_and_analyze.analyze.Graph")
    def test_analysis_skips_comparison_when_generation_has_no_references(
        self,
        mock_graph_class,
        _mock_collections,
        mock_process,
        mock_compare,
    ):
        analyze_module.start_req_analysis(self.project)

        mock_graph_class.return_value.parse.assert_called_once()
        mock_process.assert_called_once()
        mock_compare.assert_not_called()