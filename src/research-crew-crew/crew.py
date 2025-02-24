import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool
from crewai_tools import GithubSearchTool

@CrewBase
class ResearchCrewCrew():
    """ResearchCrewCrew crew"""

    # ... (rest of the file remains unchanged) 