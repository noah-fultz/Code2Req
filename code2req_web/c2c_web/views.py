import json
from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import redirect
from .utils.create_and_analyze.create_project import create_project, set_project
from c2c_web.utils.create_and_analyze.analyze import analyze
from c2c_web.utils.utils import get_project_by_id, get_all_projects, get_project_by_session
from c2c_web.utils.source_details.source_details import get_source_details
from c2c_web.utils.alignment.alignment import get_alignment
from c2c_web.utils.network.network import get_network, expand_network

def network_view(request):
    project = get_project_by_session(request)
    nodes, edges = get_network(request, project)
    return render(request, 'c2c/network.html', {'graph_data': json.dumps(nodes + edges),'projects':get_all_projects()})

def network_expand_view(request):
    project = get_project_by_session(request)
    nodes, edges = expand_network(request, project)
    return JsonResponse({"nodes": nodes,"edges": edges})


def alignment_view(request):
    project = get_project_by_session(request)
    alignment = get_alignment(project)
    context = {
        'alignment': alignment,
        'projects':get_all_projects()
    }
    return render(request, 'c2c/alignment.html', context)

def source_detail(request):
    project = get_project_by_session(request)
    artifact_id, content = get_source_details(request, project)
    return JsonResponse({
        "title": artifact_id or "Source Detail",
        "blocks": content
    })

def analysis_view(request, project_id=None):
    return render(request, 'c2c/analyze.html', {
        'project': get_project_by_id(project_id),
        'projects': get_all_projects(),
        'analysis_types': ["repository", "evaluation"],
    })

def select_project(request, project_id):
    set_project(request, project_id)
    return redirect(request.META.get('HTTP_REFERER', '/'))  

def save_project(request, project_id):
    if request.method != "POST":
        return redirect(request.META.get('HTTP_REFERER', '/'))  
    create_project(request, project_id)
    return redirect(request.META.get('HTTP_REFERER', '/'))  

def start_analysis(request, project_id, analysis_type):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)
    res = analyze(project_id, analysis_type)
    if res=="ERROR":
        return JsonResponse({'success': False, 'message': 'Unknown analysis type.'}, status=400)
    return JsonResponse({'success': True, 'message': f'{analysis_type.title()} analysis started.'})
