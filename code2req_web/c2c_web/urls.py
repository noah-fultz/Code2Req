from django.urls import path
from . import views

urlpatterns = [
    path('', views.network_view, name='network_view'),
    path('network/', views.network_view, name='network_view'),
    path('network/expand/', views.network_expand_view, name='network_expand_view'),
    path('alignment', views.alignment_view, name='alignment_view'),
    path('source-detail/', views.source_detail, name='source_detail'),
    path('analyze/', views.analysis_view, name='analysis_view'),
    path('analyze/<int:project_id>/', views.analysis_view, name='analysis_view'),
    path('save-project/<int:project_id>/', views.save_project, name='save_project'),
    path('select-project/<int:project_id>/', views.select_project, name='select_project'),
    path('start-analysis/<int:project_id>/<str:analysis_type>/', views.start_analysis, name='start_analysis'),

]