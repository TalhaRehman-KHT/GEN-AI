from pydantic import BaseModel

class BlogState(BaseModel):

    ### UserInput Fields
    topics:str=""
    audience:str="general readers"



    ### Researcher Filed
    research:str=""
    research_feedback:str=""


    ### Writer output
    draft:str=""
    draft_feedback:str=""

    ### Editor output
    final_blog:str=""

    ### MetaData
    revision_count:int=0
    published:bool=False
