from django.test import TestCase
from django.urls import reverse

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
