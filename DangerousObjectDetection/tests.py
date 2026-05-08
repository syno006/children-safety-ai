from django.test import TestCase
from django.urls import reverse
from types import SimpleNamespace

from .models import Source


class DangerousWorkspaceTests(TestCase):
	def test_workspace_loads(self):
		response = self.client.get(reverse('dangerous:index'))
		self.assertEqual(response.status_code, 200)

	def test_create_source(self):
		response = self.client.post(
			reverse('dangerous:create-source'),
			data='{"name":"Cam 1","source_type":"camera"}',
			content_type='application/json',
		)
		self.assertEqual(response.status_code, 200)
		self.assertTrue(Source.objects.filter(name='Cam 1').exists())


class TwoModelProcessingHelpersTests(TestCase):
	def test_two_model_paths_are_repo_models_based(self):
		from .views import _two_model_paths
		paths = _two_model_paths()
		self.assertIn('fire_detection', paths)
		self.assertIn('dangerous_object_i', paths)
		# Normalize path separators.
		fire = paths['fire_detection'].replace('\\', '/')
		danger = paths['dangerous_object_i'].replace('\\', '/')
		self.assertIn('/models/FireDetec/', fire)
		self.assertIn('/models/dangerousOb/', danger)

	def test_annotate_frame_respects_enabled_and_threshold(self):
		# Monkeypatch the registry-level annotator so this test doesn't
		# require OpenCV or ultralytics.
		import DangerousObjectDetection.registry as registry
		from . import views

		calls = []

		def fake_annotate(model_path, frame_bgr, *, threshold=50):
			calls.append((model_path, float(threshold)))
			return frame_bgr, [], 0.0

		orig = registry.annotate_frame_bgr_with_yolo
		registry.annotate_frame_bgr_with_yolo = fake_annotate
		try:
			source = SimpleNamespace(model_config={
				'fire_detection': {'enabled': False, 'threshold': 12},
				'dangerous_object_i': {'enabled': True, 'threshold': 77},
			})
			views._annotate_frame_with_two_models(frame_bgr=object(), source=source)  # type: ignore[arg-type]
		finally:
			registry.annotate_frame_bgr_with_yolo = orig

		self.assertEqual(len(calls), 1)
		self.assertAlmostEqual(calls[0][1], 77.0)
