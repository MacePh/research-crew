import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool
from crewai_tools import GithubSearchTool

@CrewBase
class ResearchCrewCrew():
    """ResearchCrewCrew crew"""

    def __init__(self):
        self.inputs = None  # Initialize inputs as None
        self.tasks_config = self.load_tasks_config()  # Load tasks configuration

    def load_tasks_config(self):
        """Load tasks configuration from YAML file"""
        import yaml
        with open("src/research_crew_crew/config/tasks.yaml", "r") as f:
            return yaml.safe_load(f)

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
        crew = Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

        # Run the crew and get the result
        result = crew.kickoff(inputs=self.inputs)  # Pass inputs to kickoff

        # Save the result to a file, including the topic
        if self.inputs and 'user_goal' in self.inputs:
            with open("report.md", "w", encoding="utf-8") as f:
                f.write(f"# Topic: {self.inputs['user_goal']}\n\n")
                
                # Collect outputs from each task
                for task in crew.tasks:
                    if hasattr(task, 'output') and task.config:  # Check if task has output and config
                        f.write(f"## {task.config.get('description', 'Task Description')}\n\n")
                        f.write(f"**Expected Output:** {task.config.get('expected_output', 'Expected Output')}\n\n")
                        f.write(f"**Output:**\n\n{task.output}\n\n")
                        
                        # Special formatting for the mermaid flow chart
                        if task.agent == self.flow_designer():
                            f.write("```mermaid\n")
                            f.write(task.output)
                            f.write("\n```\n\n")

        return crew
