from django.contrib import admin

from .models import Source


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
	list_display = ('id', 'name', 'owner', 'source_type', 'updated_at')
	list_filter = ('source_type',)
	search_fields = ('name',)
