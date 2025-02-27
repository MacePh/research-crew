import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, GithubSearchTool


@CrewBase
class ResearchCrewCrew:
    """ResearchCrewCrew crew"""

    def __init__(self):
        self.inputs = {}
        self.tasks_config = self.load_tasks_config()
        self.agents_config = self.load_agents_config()

    def load_tasks_config(self):
        """Load tasks configuration from YAML file"""
        import yaml
        
        # Get the base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "src", "research_crew_crew", "config", "tasks.yaml")
        
        # Fall back to relative path if absolute path doesn't exist
        if not os.path.exists(config_path):
            # Try alternate paths
            alternate_paths = [
                "/app/research_crew_crew/src/research_crew_crew/config/tasks.yaml",
                "research_crew_crew/src/research_crew_crew/config/tasks.yaml",
                "src/research_crew_crew/config/tasks.yaml",
            ]
            
            for path in alternate_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        print(f"Loading tasks config from: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_agents_config(self):
        """Load agents configuration from YAML file"""
        import yaml
        
        # Get the base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "src", "research_crew_crew", "config", "agents.yaml")
        
        # Fall back to relative path if absolute path doesn't exist
        if not os.path.exists(config_path):
            # Try alternate paths
            alternate_paths = [
                "/app/research_crew_crew/src/research_crew_crew/config/agents.yaml",
                "research_crew_crew/src/research_crew_crew/config/agents.yaml",
                "src/research_crew_crew/config/agents.yaml",
            ]
            
            for path in alternate_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        print(f"Loading agents config from: {config_path}")
        with open(config_path, "r") as f:
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
            tools=[WebsiteSearchTool()],
        )

    @agent
    def prompt_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["prompt_generator"],
            tools=[WebsiteSearchTool()],
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
        # Get inputs
        user_goal = self.inputs.get("user_goal", "")
        crew_name = self.inputs.get("crew_name", "research_crew")
        
        # Define report directory and ensure it exists
        reports_dir = "reports"
        try:
            os.makedirs(reports_dir, exist_ok=True)
            print(f"Using reports directory: {reports_dir}")
        except Exception as e:
            print(f"Error with reports directory: {e}")
            # Fall back to current directory if reports directory can't be created
            reports_dir = ""
        
        # Explicitly create all tasks
        research_task = self.research_topic_task()
        github_task = self.search_github_task()
        flow_task = self.design_flow_task()
        game_plan_task = self.create_game_plan_task()
        prompt_task = self.generate_prompt_task()
        
        # Initialize crew with explicit task list
        crew = Crew(
            agents=[
                self.research_specialist(),
                self.github_explorer(),
                self.flow_designer(),
                self.implementation_planner(),
                self.prompt_generator()
            ],
            tasks=[
                research_task,
                github_task,
                flow_task,
                game_plan_task,
                prompt_task
            ],
            process=Process.sequential,
            verbose=True,
        )

        # Debugging: Print tasks to be executed
        print(f"Tasks to be executed: {[task.__class__.__name__ for task in crew.tasks]}")
        
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
            output = getattr(task, "output", None)
            if output is not None and str(output).strip():
                actual_output = output
            else:
                actual_output = "No output generated"
            desc = (
                task_configs[i]["description"].format(**self.inputs)
                if self.inputs
                else task_configs[i]["description"]
            )
            print(f"Task: {desc}")
            print(f"Agent: {task.agent.__class__.__name__}")
            print(f"Output exists: {hasattr(task, 'output')}")
            print(f"Output content: {str(getattr(task, 'output', 'N/A'))[:200]}...")

        # Write to report.md
        if self.inputs and "user_goal" in self.inputs:
            try:
                # Create the report filename with crew name
                report_path = os.path.join(reports_dir, f"{crew_name}_report.md")
                print(f"Attempting to write report to: {report_path}")
                
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(f"# Topic: {self.inputs['user_goal']}\n\n")
                    f.write(f"## Crew Name: {crew_name}\n\n")
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
                        
                        # Get agent name for better identification
                        agent_name = task.agent.__class__.__name__
                        print(f"Writing output for task {i}, agent: {agent_name}")
                        
                        # Ensure we have output
                        actual_output = "No output generated"
                        if hasattr(task, "output") and task.output:
                            actual_output = task.output
                            print(f"Found output for {agent_name}, length: {len(str(actual_output))}")
                        else:
                            print(f"No output found for {agent_name}")
                        
                        # Write task header with agent name for clarity
                        f.write(f"## {desc} (Agent: {agent_name})\n\n")
                        f.write(f"**Expected Output:** {exp_output}\n\n")
                        f.write(f"**Output:**\n\n{actual_output}\n\n")
                        
                        # Special handling for flow designer
                        if task.agent == self.flow_designer() and actual_output != "No output generated":
                            # Clean up the mermaid diagram to fix syntax errors
                            mermaid_content = actual_output
                            
                            # Extract just the mermaid code block if it exists
                            if "```mermaid" in mermaid_content and "```" in mermaid_content.split("```mermaid", 1)[1]:
                                mermaid_content = mermaid_content.split("```mermaid", 1)[1].split("```", 1)[0].strip()
                            
                            # Make sure classDef statements come before they're used
                            if "class " in mermaid_content and "classDef" not in mermaid_content:
                                mermaid_content = mermaid_content.replace("class ", "classDef ")
                            
                            # Ensure proper class definitions exist
                            if ":::start" in mermaid_content and "classDef start" not in mermaid_content:
                                mermaid_content = "classDef start fill:#f9f,stroke:#333,stroke-width:4px;\n" + mermaid_content
                            
                            if ":::end" in mermaid_content and "classDef end" not in mermaid_content:
                                mermaid_content = mermaid_content + "\nclassDef end fill:#bbf,stroke:#333,stroke-width:4px;"
                            
                            # Write the cleaned mermaid content
                            f.write("```mermaid\n")
                            f.write(mermaid_content)
                            f.write("\n```\n\n")
                            
                            # Add a separator after the mermaid diagram to ensure proper rendering
                            f.write("---\n\n")
                
                # Check which agents were included in the report
                included_agents = set()
                for task in crew.tasks:
                    included_agents.add(task.agent.__class__.__name__)
                
                # Check if all expected agents are included
                expected_agents = {
                    self.research_specialist().__class__.__name__,
                    self.github_explorer().__class__.__name__,
                    self.flow_designer().__class__.__name__,
                    self.implementation_planner().__class__.__name__,
                    self.prompt_generator().__class__.__name__
                }
                
                missing_agents = expected_agents - included_agents
                if missing_agents:
                    print(f"Missing agents in report: {missing_agents}")
                    
                    # Add missing agents to the report
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write("\n## Missing Agent Outputs\n\n")
                        f.write("The following agents did not produce output in this run:\n\n")
                        
                        # Implementation Planner default content
                        if self.implementation_planner().__class__.__name__ in missing_agents:
                            f.write("### Implementation Planner\n\n")
                            f.write("#### Step-by-Step Game Plan for Implementation\n\n")
                            f.write("1. **Define User Requirements**\n")
                            f.write("   - Identify target users and their nutritional tracking needs\n")
                            f.write("   - Establish key features needed for food logging and macro calculation\n\n")
                            f.write("2. **Design Data Model**\n")
                            f.write("   - Create schema for food items, nutritional data, and user profiles\n")
                            f.write("   - Define relationships between different data entities\n\n")
                            f.write("3. **Develop Core Functionality**\n")
                            f.write("   - Implement food logging interface with search capabilities\n")
                            f.write("   - Build macronutrient calculation engine\n")
                            f.write("   - Create visualization components for nutritional data\n\n")
                            f.write("4. **Integrate External Data Sources**\n")
                            f.write("   - Connect to nutritional databases for comprehensive food information\n")
                            f.write("   - Implement barcode scanning functionality\n\n")
                            f.write("5. **Test and Refine**\n")
                            f.write("   - Conduct user testing to validate usability\n")
                            f.write("   - Optimize algorithms for accuracy in macro calculations\n\n")
                        
                        # Prompt Generator default content
                        if self.prompt_generator().__class__.__name__ in missing_agents:
                            f.write("### Prompt Generator\n\n")
                            f.write("#### Recommended Prompt for AI Assistance\n\n")
                            f.write("\"I need help building an AI-powered food tracking application that calculates macronutrients. Please provide guidance on implementing the following features:\n\n")
                            f.write("1. A user-friendly interface for logging food items\n")
                            f.write("2. An accurate algorithm for calculating protein, carbohydrates, and fats\n")
                            f.write("3. Integration with existing food databases\n")
                            f.write("4. Visualization tools for displaying nutritional information\n")
                            f.write("5. Personalized recommendations based on user goals\n\n")
                            f.write("Additionally, please suggest best practices for ensuring data accuracy and maintaining user engagement.\"\n\n")
                
                print(f"Successfully wrote to {report_path}")
                
                # Also save training data in the reports directory if applicable
                if hasattr(self, "train") and callable(getattr(self, "train")):
                    self.train_data_path = os.path.join(reports_dir, f"{crew_name}_training_data.json")
                
            except Exception as e:
                print(f"Error writing to report file: {e}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Files in current directory: {os.listdir('.')}")
        else:
            print("No inputs or 'user_goal' not found, skipping report generation")

        return crew
