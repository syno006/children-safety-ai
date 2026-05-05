from django.conf import settings
from django.db import models


class Source(models.Model):
	SOURCE_TYPES = [
		('image', 'Image'),
		('video', 'Video'),
		('camera', 'Camera'),
	]

	name = models.CharField(max_length=120)
	owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
	source_type = models.CharField(max_length=20, choices=SOURCE_TYPES)
	layout = models.JSONField(default=dict, blank=True)
	model_config = models.JSONField(default=dict, blank=True)
	camera_config = models.JSONField(default=dict, blank=True)
	last_results = models.JSONField(default=dict, blank=True)
	last_media_path = models.CharField(max_length=255, blank=True)
	last_media_name = models.CharField(max_length=255, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ['created_at']

	def __str__(self):
		return f"{self.name} ({self.source_type})"
