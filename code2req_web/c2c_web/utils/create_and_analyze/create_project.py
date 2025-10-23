from c2c_web.models import Project, EvaluationFile
from django.shortcuts import get_object_or_404, redirect
from django.conf import settings
import os
import requests
from git import Repo, GitCommandError
import zipfile
import tempfile
from pathlib import Path
import shutil

def clone_or_pull_repo(git_url, clone_dir):
    if os.path.exists(clone_dir) and os.listdir(clone_dir):
        # Repo already exists, pull latest changes
        print(f"Pulling latest changes into {clone_dir}...")
        try:
            repo = Repo(clone_dir)
            origin = repo.remotes.origin
            origin.pull()
        except GitCommandError as e:
            print(f"Error during pull: {e}")
            raise
    else:
        # Directory empty or doesn't exist, clone it
        print(f"Cloning {git_url} into {clone_dir}...")
        try:
            Repo.clone_from(git_url, clone_dir)
        except GitCommandError as e:
            print(f"Error during clone: {e}")
            raise



def handle_zip_upload(zip_file, project_name):
    target_dir = Path("data") / project_name / "repo"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, zip_file.name)
        with open(zip_path, "wb") as f:
            for chunk in zip_file.chunks():
                f.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        extracted_root = Path(tmpdir)
        contents = list(extracted_root.iterdir())

        # If the ZIP has a top-level folder, dive in
        if len(contents) == 1 and contents[0].is_dir():
            contents = list(contents[0].iterdir())

        for item in contents:
            shutil.move(str(item), str(target_dir))

    return target_dir


import re
import requests

SAFE_RULESETS = {
    "empty", "rdfsplus-optimized", "rdfsplus", "owl-horst-optimized", "owl2-rl-optimized"
}

def _repo_exists(repo_id: str, base_url: str, auth=None, timeout: int = 15) -> bool:
    r = requests.get(f"{base_url.rstrip('/')}/rest/repositories/{repo_id}",
                     auth=auth, timeout=timeout)
    return r.status_code == 200

def _delete_repo(repo_id: str, base_url: str, auth=None, timeout: int = 30) -> None:
    requests.delete(f"{base_url.rstrip('/')}/rest/repositories/{repo_id}",
                    auth=auth, timeout=timeout)

def create_and_register_repo(
    repo_id: str,
    base_url: str = "http://graphdb:7200",  # from the web container use the service name
    label: str | None = None,
    ruleset: str = "rdfsplus-optimized",
    auth=None,
    timeout: int = 30,
    overwrite: bool = False,                 # <- add this
) -> tuple[bool, str]:

    # 1) basic input safety
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_id):
        return False, "repo_id must match [A-Za-z0-9_.-]+"
    if ruleset not in SAFE_RULESETS:
        return False, f"ruleset must be one of {sorted(SAFE_RULESETS)}"

    # 2) overwrite if requested
    if overwrite and _repo_exists(repo_id, base_url, auth, timeout=timeout):
        _delete_repo(repo_id, base_url, auth, timeout=timeout)

    # 3) minimal Sail repo config (no OBDA anywhere)
    ttl = f"""@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix rep:   <http://www.openrdf.org/config/repository#> .
        @prefix sr:    <http://www.openrdf.org/config/repository/sail#> .
        @prefix sail:  <http://www.openrdf.org/config/sail#> .
        @prefix graphdb: <http://www.ontotext.com/config/graphdb#> .

        <#repo-{repo_id}> a rep:Repository ;
        rep:repositoryID "{repo_id}" ;
        rep:repositoryImpl [
            rep:repositoryType "graphdb:SailRepository" ;
            sr:sailImpl [
            sail:sailType "graphdb:Sail" ;
            graphdb:ruleset "{ruleset}" ;
            graphdb:enable-literal-index "true" ;
            graphdb:disable-sameAs "true"
            ]
        ] ;
        rdfs:label "{(label or repo_id).replace('"','')}" .
        """

    # 4) POST the config (let requests set multipart boundary)
    endpoint = base_url.rstrip("/") + "/rest/repositories"
    files = {"config": ("config.ttl", ttl.encode("utf-8"), "text/turtle")}
    resp = requests.post(endpoint, files=files, auth=auth, timeout=timeout)

    if resp.status_code in (200, 201):
        return True, f"repo '{repo_id}' created"
    if resp.status_code == 409:
        return True, f"repo '{repo_id}' already exists"
    return False, f"{resp.status_code}: {resp.text}"


def create_project(request, project_id):
    if project_id == 0:
        project = Project()
    else:
        project = get_object_or_404(Project, id=project_id)
    project.name = request.POST.get('project_name')
    project_path = os.path.join("data/", project.name)
    os.makedirs(project_path, exist_ok=True)
    comparisons_path = os.path.join(project_path, "comparisons")
    os.makedirs(comparisons_path, exist_ok=True)
    ttl_files_path = os.path.join(project_path, "ttl-files")
    os.makedirs(ttl_files_path, exist_ok=True)
    create_and_register_repo(project.name, settings.SPARQL_ENDPOINT.replace("/repositories/",""), project_path+"/repo-config.ttl")
    
    github_url = request.POST.get("github_repo")
    source_zip = request.FILES.get("source_zip")
    if project.github_repo=="" and github_url:
        project.github_repo = request.POST.get('github_repo')
        clone_or_pull_repo(project.github_repo, "data/"+project.name+"/repo/")
    elif source_zip:
        project.github_repo = "data/"+project.name+"/repo/"
        handle_zip_upload(source_zip, project.name)
    project.clusters = request.POST.get('clusters')
    project.top_k = request.POST.get('top_k')
    project.max_tokens = request.POST.get('max_tokens')
    project.model = request.POST.get('model')
    project.use_rag     = "use_rag" in request.POST
    project.use_feature = "use_feature" in request.POST
    project.use_code    = "use_code" in request.POST


    eval_path = os.path.join(project_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)
    project.save()
    
    for key in request.FILES:
        if key.startswith('eval_file_'):
            index = key.split('_')[-1]
            name_key = f'eval_name_{index}'
            custom_name = request.POST.get(name_key, f'eval_{index}')
            uploaded_file = request.FILES[key]

            # Create a safe filename
            safe_filename = f"{custom_name}.csv"
            full_path = os.path.join(eval_path, safe_filename)

            # Save to disk
            with open(full_path, 'wb+') as dest:
                for chunk in uploaded_file.chunks():
                    dest.write(chunk)

            # Save metadata
            EvaluationFile.objects.create(
                project=project,
                name=custom_name,
                path=full_path
            )
    files_to_delete = request.POST.get("files_to_delete", "")
    if files_to_delete:
        ids = [int(i) for i in files_to_delete.split(",") if i.strip().isdigit()]
        for file in EvaluationFile.objects.filter(id__in=ids, project=project):
            # Remove from disk if needed
            if os.path.exists(file.path):
                os.remove(file.path)
            file.delete()

    project.save()


def set_project(request, project_id):
    try:
        Project.objects.get(id=project_id)  # Validate
        request.session['selected_project_id'] = project_id
    except Project.DoesNotExist:
        request.session['selected_project_id'] = 1  # Default fallback
