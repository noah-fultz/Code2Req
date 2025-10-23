from .models import Project

def project_context(request):
    projects = Project.objects.all().order_by('name')

    if not projects.exists():
        # No projects in database
        return {
            'current_project': None,
            'projects': projects,
        }

    project_id = request.session.get('selected_project_id')

    if project_id:
        try:
            current_project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            current_project = projects.first()  # Fallback to first available
            request.session['selected_project_id'] = current_project.id
    else:
        current_project = projects.first()
        request.session['selected_project_id'] = current_project.id

    return {
        'current_project': current_project,
        'projects': projects,
    }
