from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=255)
    github_repo = models.URLField()
    requirement_file = models.CharField(max_length=500, null=True, blank=True)
    
    clusters = models.IntegerField(default=10)
    top_k = models.IntegerField(default=5)
    max_tokens = models.IntegerField(default=400000)
    model = models.CharField(max_length=500, null=True, blank=True, default="gpt-5")
    use_rag = models.BooleanField(default=True)
    use_feature = models.BooleanField(default=True)
    use_code = models.BooleanField(default=True)

    repository_analyzed = models.BooleanField(default=False)
    requirements_analyzed = models.BooleanField(default=False)

    lines_of_code = models.IntegerField(default=0)
    functions = models.IntegerField(default=0)
    classes = models.IntegerField(default=0)
    snippets = models.IntegerField(default=0)
    imports = models.IntegerField(default=0)
    artifacts = models.IntegerField(default=0)
    codefiles = models.IntegerField(default=0)
    features = models.IntegerField(default=0)
    code_tokens = models.IntegerField(default=0)
    feature_tokens = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class EvaluationFile(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='evaluation_files')
    name = models.CharField(max_length=255)
    path = models.TextField() 
    f1_score = models.FloatField(null=True, blank=True)
    num_linked_to_feature = models.IntegerField(default=0)
    num_units = models.IntegerField(default=0)

class EvaluationUnit(models.Model):
    file = models.ForeignKey(EvaluationFile, on_delete=models.CASCADE, related_name='units')
    identifier = models.CharField(max_length=100)
    description = models.TextField()
    linked_feature_ids = models.JSONField(default=list)  # Use ArrayField if Postgres-only
    scores = models.JSONField(default=list)
    score = models.FloatField(null=True, blank=True)
    transformed_description = models.TextField(blank=True)
    linked_code = models.JSONField(default=list)  

