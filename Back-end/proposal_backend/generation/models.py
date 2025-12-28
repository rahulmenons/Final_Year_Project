from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError


class Document(models.Model):
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=10)
    upload_date = models.DateTimeField(auto_now_add=True)
    content_preview = models.TextField(blank=True, null=True)
    processed = models.BooleanField(default=False)
    summary = models.TextField(blank=True, null=True)

    # --- NEW: structured RFP fields for scoring ---
    rfp_budget = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Expected project budget in INR (e.g. 800000 for ₹8L)",
    )
    rfp_emd = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Earnest Money Deposit (EMD) in INR if present in RFP",
    )
    rfp_timeline_weeks = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Expected duration in weeks",
    )
    # Additional timeline-related days that Gemini may extract
    no_of_days_for_analysis = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="No. of days allocated for analysis (if mentioned)"
    )
    no_of_days_for_submission = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="No. of days allowed for submission (if mentioned)"
    )

    rfp_team_size_required = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Number of people required by client",
    )

    # store raw metadata JSON returned by Gemini (for audit/debug)
    rfp_metadata = models.JSONField(null=True, blank=True, help_text="Raw extracted RFP metadata JSON (from Gemini)")

    # store extraction confidence & notes separately for easy queries
    extraction_confidence = models.CharField(
        max_length=10,
        null=True,
        blank=True,
        help_text='Confidence level reported by extractor (high|medium|low)'
    )
    extraction_notes = models.TextField(null=True, blank=True, help_text="Short notes from metadata extractor")

    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("ACCEPTED", "Accepted"),
        ("REJECTED", "Rejected"),
        ("REVIEW", "Needs Review"),
    ]
    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default="PENDING",
    )

    class Meta:
        ordering = ['-upload_date']
    
    def __str__(self):
        return self.filename


class Keyword(models.Model):
    keyword = models.CharField(max_length=255, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['keyword']
    
    def __str__(self):
        return self.keyword


class DocumentKeyword(models.Model):
    document = models.ForeignKey(
        Document, 
        on_delete=models.CASCADE, 
        related_name='keywords'
    )
    keyword = models.ForeignKey(
        Keyword, 
        on_delete=models.CASCADE, 
        related_name='documents'
    )
    relevance_score = models.FloatField()
    
    class Meta:
        unique_together = ('document', 'keyword')
        ordering = ['-relevance_score']
    
    def __str__(self):
        return f"{self.document.filename} - {self.keyword.keyword}"


# --- Company capability (singleton-ish) ---

class CompanyCapability(models.Model):
    """
    Company fixed capabilities used to evaluate incoming RFPs.
    Only ONE row should exist.
    """

    # already existing fields
    tech_keywords = ArrayField(models.CharField(max_length=100), default=list)

    min_budget = models.PositiveIntegerField()
    max_budget = models.PositiveIntegerField()

    expected_emd_in_inr = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Expected EMD range for projects (INR)"
    )

    min_timeline_weeks = models.PositiveIntegerField()
    max_timeline_weeks = models.PositiveIntegerField()

    max_team_size = models.PositiveIntegerField()
    
    # -------------------------------------------------------
    # NEW FIELDS — matching Gemini extraction JSON structure
    # -------------------------------------------------------

    expected_timeline_weeks = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Expected timeline for typical projects in weeks"
    )

    expected_no_of_days_for_analysis = models.PositiveIntegerField(
        null=True, blank=True
    )

    expected_no_of_days_for_submission = models.PositiveIntegerField(
        null=True, blank=True
    )

    # meta
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Company Capability Profile"

    def clean(self):
        """Ensure only ONE capability row exists."""
        if not self.pk and CompanyCapability.objects.exists():
            raise ValidationError("Only one CompanyCapability instance is allowed.")


# --- Evaluation per document ---

class RFPEvaluation(models.Model):
    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name="evaluation",
    )

    technical_fit_score = models.FloatField(default=0.0)
    budget_fit_score = models.FloatField(default=0.0)
    timeline_fit_score = models.FloatField(default=0.0)
    capacity_fit_score = models.FloatField(default=0.0)
    overall_fit_score = models.FloatField(default=0.0)

    DECISION_CHOICES = [
        ("ACCEPT", "Accept"),
        ("REJECT", "Reject"),
        ("REVIEW", "Review"),
    ]
    decision = models.CharField(
        max_length=10,
        choices=DECISION_CHOICES,
        default="REVIEW",
    )

    reasoning = models.TextField(
        blank=True,
        help_text="Short explanation of why this decision was taken.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Evaluation for {self.document.filename} ({self.decision})"
