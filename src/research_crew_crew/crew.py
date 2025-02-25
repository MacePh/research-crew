import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, GithubSearchTool


@CrewBase
class ResearchCrewCrew:
    """ResearchCrewCrew crew"""

    def __init__(self):
        self.inputs = None
        self.tasks_config = self.load_tasks_config()
        self.agents_config = self.load_agents_config()

    def load_tasks_config(self):
        """Load tasks configuration from YAML file"""
        import yaml

        with open("src/research_crew_crew/config/tasks.yaml", "r") as f:
            return yaml.safe_load(f)

    def load_agents_config(self):
        """Load agents configuration from YAML file"""
        import yaml

        with open("src/research_crew_crew/config/agents.yaml", "r") as f:
            return yaml.safe_load(f)

    @agent
    def research_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["research_specialist"],
            tools=[WebsiteSearchTool()],
        )

    @agent
    def github_explorer(self) -> Agent:
        return Agent(
            config=self.agents_config["github_explorer"],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["code", "repositories"],
                )
            ],
        )

    @agent
    def flow_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["flow_designer"],
            tools=[],
        )

    @agent
    def implementation_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["implementation_planner"],
            tools=[],
        )

    @agent
    def prompt_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["prompt_generator"],
            tools=[],
        )

    @task
    def research_topic_task(self) -> Task:
        config = self.tasks_config["research_topic_task"]
        return Task(
            description=config["description"].format(**self.inputs)
            if self.inputs
            else config["description"],
            expected_output=config["expected_output"].format(**self.inputs)
            if self.inputs
            else config["expected_output"],
            tools=[WebsiteSearchTool()],
            agent=self.research_specialist(),
        )

    @task
    def search_github_task(self) -> Task:
        config = self.tasks_config["search_github_task"]
        return Task(
            description=config["description"].format(**self.inputs)
            if self.inputs
            else config["description"],
            expected_output=config["expected_output"].format(**self.inputs)
            if self.inputs
            else config["expected_output"],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["code", "repositories"],
                )
            ],
            agent=self.github_explorer(),
        )

    @task
    def design_flow_task(self) -> Task:
        config = self.tasks_config["design_flow_task"]
        return Task(
            description=config["description"].format(**self.inputs)
            if self.inputs
            else config["description"],
            expected_output=config["expected_output"].format(**self.inputs)
            if self.inputs
            else config["expected_output"],
            tools=[],
            agent=self.flow_designer(),
        )

    @task
    def create_game_plan_task(self) -> Task:
        config = self.tasks_config["create_game_plan_task"]
        return Task(
            description=config["description"].format(**self.inputs)
            if self.inputs
            else config["description"],
            expected_output=config["expected_output"].format(**self.inputs)
            if self.inputs
            else config["expected_output"],
            tools=[],
            agent=self.implementation_planner(),
        )

    @task
    def generate_prompt_task(self) -> Task:
        config = self.tasks_config["generate_prompt_task"]
        return Task(
            description=config["description"].format(**self.inputs)
            if self.inputs
            else config["description"],
            expected_output=config["expected_output"].format(**self.inputs)
            if self.inputs
            else config["expected_output"],
            tools=[],
            agent=self.prompt_generator(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchCrewCrew crew"""
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

        # Debugging: Print inputs
        print(f"Inputs: {self.inputs}")

        # Run the crew and get the result
        result = crew.kickoff(inputs=self.inputs)
        print(f"Crew result: {result}")

        # Debugging: Inspect task outputs
        task_configs = [
            self.tasks_config["research_topic_task"],
            self.tasks_config["search_github_task"],
            self.tasks_config["design_flow_task"],
            self.tasks_config["create_game_plan_task"],
            self.tasks_config["generate_prompt_task"],
        ]
        for i, task in enumerate(crew.tasks):
            output = getattr(task, "output", "No output")
            desc = (
                task_configs[i]["description"].format(**self.inputs)
                if self.inputs
                else task_configs[i]["description"]
            )
            print(f"Task: {desc}, Output: {output}")

        # Write to report.md
        if self.inputs and "user_goal" in self.inputs:
            try:
                with open("report.md", "w", encoding="utf-8") as f:
                    f.write(f"# Topic: {self.inputs['user_goal']}\n\n")
                    for i, task in enumerate(crew.tasks):
                        desc = (
                            task_configs[i]["description"].format(**self.inputs)
                            if self.inputs
                            else task_configs[i]["description"]
                        )
                        exp_output = (
                            task_configs[i]["expected_output"].format(**self.inputs)
                            if self.inputs
                            else task_configs[i]["expected_output"]
                        )
                        actual_output = (
                            task.output
                            if hasattr(task, "output") and task.output
                            else "No output generated"
                        )
                        f.write(f"## {desc}\n\n")
                        f.write(f"**Expected Output:** {exp_output}\n\n")
                        f.write(f"**Output:**\n\n{actual_output}\n\n")
                        if task.agent == self.flow_designer():
                            f.write("```mermaid\n")
                            f.write(actual_output)
                            f.write("\n```\n\n")
                print("Successfully wrote to report.md")
            except Exception as e:
                print(f"Error writing to report.md: {e}")
        else:
            print("No inputs or 'user_goal' not found, skipping report generation")

        return crew
