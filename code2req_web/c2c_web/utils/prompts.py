from pydantic import BaseModel, Field
from typing import Literal, List


####################
# Extract features #
####################

class Feature(BaseModel):
    name: str = Field(description='Software feature name')
    description: str = Field(description='Software feature description')

##############################
# Extract features from code #
##############################

def get_tag_features_sys_prompt(type):
    if type=="Code":
        return """You are an expert at summarizing groups of related software code into high-level software FEATURES.
        You will be given a dictionary that maps code artifact URIs to their individual code. 
        These code artifacts are all working together to accomplish a shared TASK or a closely related set of responsibilities. 
        Your job is to ANALYZE ALL pieces of code together and generate a high-level FEATURE that reflect what the group is accomplishing.
        Feature description should be written in a direct and clear language, describing what is being achieved by the group. Never start a description with 'this group' or 'this feature'.
        Do not guess or interpret the meaning of acronyms. Only explain an acronym if its full form is explicitly provided. Otherwise, use the acronym as it appears, without expanding it.

        Here is an example for guidance:
        Given these code artifact descriptions: {
        'http://example.org/repo/FunCoup/data/models.py#Proteome': 
            'class Proteome(models.Model): species = models.ForeignKey(Species, on_delete=models.CASCADE) version = models.CharField(max_length=100) class Meta: indexes = [models.Index(fields=['species','version'])]'
        'http://example.org/repo/FunCoup/data/models.py#Species': 
           'class Species(models.Model): tax_id = models.CharField(max_length=20) species_name = models.CharField(max_length=200) common_name = models.CharField(max_length=200) class Meta: indexes = [models.Index(fields=['tax_id'])]' 
        }
        This Feature should be extracted: {
        name='Species and Proteome Data Model Definitions' 
        description='Model definitions for Species and Proteome. Proteome has a foreign key to Species, and multiple indexes are applied to optimize search performance.'
        }
        Note that Feature name and descriptions should ALWAYS be in ENGLISH!
        """
    elif type=="Features":
        return """You are an expert at summarizing groups of software descriptions into higher-level software FEATURE SUMMARIES.

        You will be given a dictionary that maps feature URIs to their individual descriptions. 
        These software features are all working together to accomplish a shared TASK or a closely related set of responsibilities. 
        Your job is to ANALYZE ALL descriptions together and generate a higher-level FEATURE summary that reflect what the GROUP is accomplishing.
        Feature description should be written in a direct and clear language, describing what is being achieved by the group. Never start a description with 'this group' or 'this feature'.
        Do not guess or interpret the meaning of acronyms. Only explain an acronym if its full form is explicitly provided. Otherwise, use the acronym as it appears, without expanding it.

        Here is an example for guidance:
        Given these software feature descriptions: {
        'http://example.org/repo#Protein_Identification_and_Mapping_5d3698ea45_feature': 
            'Mapping and identification of Proteins. Defines models for Protein and IdMapping. Protein has a foreign key to Species. These are used to obtain the id of a protein, either directly using the protein, or through an IdMapping.', 
        'http://example.org/repo#Species_and_Proteome_Data_Model_Definitions_bc99090d6a_feature': 
            'Model definitions for Species and Proteome. Proteome has a foreign key to Species, and multiple indexes are applied to optimize search performance.'
        }
        This Feature should be extracted: {
        name='Protein ID extraction' 
        description='Extracts the ID of a protein, either directly as a Protein, or through IdMapping. The Protein has a relation to Proteome which in turn has a relation to Species.'
        }
        Note that Feature name and descriptions should ALWAYS be in ENGLISH!
        """
def get_tag_features_user_prompt(desc):
    return f"Now summarize these artifacts into a higher-level software feature with a name and description in English: {desc}"


class Requirement(BaseModel):
    description: str = Field(description="Requirement description in given format")

class RequirementList(BaseModel):
    requirements: List[Requirement] = Field(description="List of requirements derived from the features")

####################
# Compare features #
####################

def get_generate_req_from_feature_sys_prompt(example_1, example_2, example_3, format, use_rag):
    if use_rag==False:
        return f"""You are an expert at making functional requirements from software features. 
                You will be given software features for a system. Your task is to generate functional REQUIREMENTS from the features.
                ALL generated requirement descriptions MUST adhere to this format:
                {format}
                Here are some examples illustrating how the format should be used:
                Example 1: {example_1}  
                Example 2: {example_2}
                Example 3: {example_3} 
        """
    else:
        return f"""You are an expert at making functional requirements from software features. 
            You will be given software features for a system. Your task is to generate functional REQUIREMENTS from the features.
            Use your TOOLS to get information on existing requirements that you can use as reference when formulating new ones.
            ALL generated requirement descriptions MUST adhere to this format:
            {format}
            Here are some examples illustrating how the format should be used:
            Example 1: {example_1}  
            Example 2: {example_2}
            Example 3: {example_3} 
        """

def get_generate_req_from_feature_user_prompt(use_rag):
    if use_rag==False:
        return f"""Make functional requirements from the given features
            Make sure that ALL requirement descriptions follow the style given in the examples! """
    else:
        return f"""Make functional requirements from the given features
            Make sure that ALL requirement descriptions follow the style given in the examples!
            REMEMBER TO USE YOUR TOOLS TO GET EXISTING REQUIREMENTS for reference"""
    

def get_generate_req_from_code_sys_prompt(example_1, example_2, example_3, format, use_rag):
    if use_rag==False:
        return f"""You are an expert at making functional requirements from software code. 
                You will be given software code for a system. Your task is to generate functional REQUIREMENTS for the code.
                ALL generated requirement descriptions MUST adhere to this format:
                {format}
                Here are some examples illustrating how the format should be used:
                Example 1: {example_1}  
                Example 2: {example_2}
                Example 3: {example_3} 
            """
    else:
        return f"""You are an expert at making functional requirements from software code. 
                You will be given software code for a system. Your task is to generate functional REQUIREMENTS for the code.
                Use your TOOLS to get information on existing requirements that you can use as reference when formulating new ones.
                ALL generated requirement descriptions MUST adhere to this format:
                {format}
                Here are some examples illustrating how the format should be used:
                Example 1: {example_1}  
                Example 2: {example_2}
                Example 3: {example_3} 
            """

def get_generate_req_from_code_user_prompt(use_rag):
    if use_rag==False:
        return """Make functional requirements for the given software code. 
            Make sure that ALL requirement descriptions follow the style given in the examples!"""
    else:
        return """Make functional requirements for the given software code. 
            Make sure that ALL requirement descriptions follow the style given in the examples!
            REMEMBER TO USE YOUR TOOLS TO GET EXISTING REQUIREMENTS for reference"""


tools_requirements=[{
        'type': 'function',
        'function': {
            'name': 'get_software_requirement',
            'description': f"""Obtain existing requirements for the system to use for reference when formulating new ones""",
            'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                'type': 'string',
                'description': 'Keywords for the requirements to be generated.',
                },
            },
            'required': ['query'],
            },
        },
    }
]
