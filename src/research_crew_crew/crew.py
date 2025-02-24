import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool
from crewai_tools import GithubSearchTool

@CrewBase
class ResearchCrewCrew():
    """ResearchCrewCrew crew"""

    @agent
    def research_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['research_specialist'],
            tools=[WebsiteSearchTool()],
        )

    @agent
    def github_explorer(self) -> Agent:
        return Agent(
            config=self.agents_config['github_explorer'],
            tools=[GithubSearchTool(gh_token=os.getenv('GITHUB_TOKEN'), content_types=["code", "repositories"])]
        )

    @agent
    def flow_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['flow_designer'],
            tools=[],
        )

    @agent
    def implementation_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['implementation_planner'],
            tools=[],
        )

    @agent
    def prompt_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['prompt_generator'],
            tools=[],
        )


    @task
    def research_topic_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_topic_task'],
            tools=[WebsiteSearchTool()],
        )

    @task
    def search_github_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_github_task'],
            tools=[GithubSearchTool(
                gh_token=os.getenv('GITHUB_TOKEN'),
                content_types=["code", "repositories"]
            )],
        )

    @task
    def design_flow_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_flow_task'],
            tools=[],
        )

    @task
    def create_game_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_game_plan_task'],
            tools=[],
        )

    @task
    def generate_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_prompt_task'],
            tools=[],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the ResearchCrewCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
