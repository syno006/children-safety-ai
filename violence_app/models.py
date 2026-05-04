from django.db import models


class AnalysisRecord(models.Model):
    RISK_CHOICES = [('LOW', 'Low'), ('MEDIUM', 'Medium'), ('HIGH', 'High')]

    original_filename = models.CharField(max_length=255)
    stored_filename   = models.CharField(max_length=255)
    label             = models.CharField(max_length=20)   # Violent | NonViolent
    prob_violent      = models.FloatField()
    prob_safe         = models.FloatField()
    confidence        = models.FloatField()
    risk_level        = models.CharField(max_length=10, choices=RISK_CHOICES, default='LOW')
    alert             = models.BooleanField(default=False)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.risk_level}] {self.original_filename} — {self.label}"

    @property
    def risk_color(self):
        return {'HIGH': '#e63946', 'MEDIUM': '#f4a261', 'LOW': '#2dc653'}.get(self.risk_level, '#aaa')

    @property
    def risk_icon(self):
        return {'HIGH': '🚨', 'MEDIUM': '⚠️', 'LOW': '✅'}.get(self.risk_level, '•')
