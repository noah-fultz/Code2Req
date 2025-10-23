import re
from django import template

register = template.Library()


@register.filter
def feature_shortname(value):
    """
    Extracts a short readable name from a URI.
    
    - For inputs like: '...#transferNetworks_237b947b41_feature' → 'transferNetworks'
    - For inputs like: '.../anubix.py#sample_crosstalk' → 'sample_crosstalk'
    """
    # Match the special _feature format
    match = re.search(r'#(.*?)(?:_[a-f0-9]{10,}|)_feature$', value)
    if match:
        return match.group(1)

    # Match any general fragment after #
    fragment = value.split("#")[-1] if "#" in value else None
    if fragment:
        return fragment

    return value  # fallback


@register.filter
def get_analysis_status(project, analysis_type):
    if analysis_type == 'repository':
        return project.repository_analyzed
    elif analysis_type == 'evaluation':
        return project.requirements_analyzed
    return False

@register.filter
def mul(value, arg):
    """Multiply the arg and the value."""
    return int(value) * int(arg)