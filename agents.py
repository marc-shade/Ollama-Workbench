# agents.py
import json
import logging
from typing import List, Dict, Tuple, Callable, Any
from projects import Task
from search_libraries import duckduckgo_search, google_search, serpapi_search, serper_search, bing_search
import ollama
import re
import spacy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from openai_utils import call_openai_api, OPENAI_MODELS
from groq_utils import call_groq_api, GROQ_MODELS

# Set up logging
logging.basicConfig(filename='agents.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

class Agent:
    def __init__(self, name: str, capabilities: List[str], prompts: Dict[str, str], model: str = None, **kwargs):
        self.name = name
        self.capabilities = capabilities
        self.prompts = prompts
        self.model = model
        self.settings = kwargs

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def cancel_task(self, task: Task):
        """
        Cancels the execution of a task.

        Args:
            task: The task to cancel.
        """
        print(f"Canceling task: {task.name}")
        # TODO: Implement agent-specific cancellation logic

class SearchAgent(Agent):
    def __init__(self, name: str, model: str, search_function: Callable, api_key: str = None, cse_id: str = None, prompt: str = None, role: str = None):
        super().__init__(name=name, capabilities=["web_search"], prompts={}, model=model)
        self.search_function = search_function
        self.api_key = api_key
        self.cse_id = cse_id
        self.prompt = prompt
        self.role = role

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        try:
            if self.search_function == google_search:
                if self.api_key and self.cse_id:
                    return self.search_function(query, self.api_key, self.cse_id, num_results)
            elif self.search_function in [serpapi_search, serper_search, bing_search]:
                if self.api_key:
                    return self.search_function(query, self.api_key, num_results)
            else:
                return self.search_function(query, num_results)
        except Exception as e:
            logging.error(f"Search error for agent {self.name}: {str(e)}")
        return []

    def generate_summary(self, query: str, search_results: List[Dict]) -> str:
        formatted_results = ""
        for i, result in enumerate(search_results):
            formatted_results += f"[{i+1}] {result['title']}: {result['url']}\n"

        prompt = f"""{self.prompt}

        Here are some relevant search results:
        {formatted_results}

        Generate a detailed summary of the search results, addressing the following query:
        "{query}"

        Include relevant information from the search results and cite your sources using [n] notation.
        """

        try:
            if self.model in OPENAI_MODELS:
                response = call_openai_api(self.model, [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1000, openai_api_key=self.api_key)
            elif self.model in GROQ_MODELS:
                response = call_groq_api(self.model, prompt, temperature=0.7, max_tokens=1000, groq_api_key=self.api_key)
            else:
                response = ollama.generate(model=self.model, prompt=prompt)
                response = response['response']
            return response
        except Exception as e:
            logging.error(f"Summary generation error for agent {self.name}: {str(e)}")
            return f"Error generating summary: {str(e)}"

class SearchManager(Agent):
    def __init__(self, name: str, model: str, temperature: float, max_tokens: int, api_keys: Dict[str, str]):
        capabilities = ["research_management"]
        prompts = {
            "agent_creation": "Create specialized research agents based on the given request.",
            "report_compilation": "Compile a comprehensive report based on agent summaries."
        }
        super().__init__(name=name, capabilities=capabilities, prompts=prompts, model=model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.search_agents = {}
        self.api_keys = api_keys

    def create_search_agents(self, user_request: str, agent_model: str):
        search_libraries = {
            "duckduckgo": duckduckgo_search,
            "google": google_search,
            "serpapi": serpapi_search,
            "serper": serper_search,
            "bing": bing_search
        }

        agent_definition_prompt = f"""
        Based on the following research request, create a team of specialized researchers:
        "{user_request}"

        Generate a JSON object defining 5 research agents, each specializing in a different aspect of the request.
        The JSON object should have the following structure:

        {{
          "agents": [
            {{
              "name": "Name",
              "role": "Specific Research Focus",
              "library": "Search Library",
              "prompt": "Detailed instructions for the agent's research focus"
            }},
            // ... more agents
          ]
        }}

        Available search libraries: {', '.join(search_libraries.keys())}
        Ensure each agent has a unique role and uses a different search library.
        """

        try:
            if self.model in OPENAI_MODELS:
                response = call_openai_api(self.model, [{"role": "user", "content": agent_definition_prompt}], temperature=self.temperature, max_tokens=self.max_tokens, openai_api_key=self.api_keys.get("openai_api_key"))
            elif self.model in GROQ_MODELS:
                response = call_groq_api(self.model, agent_definition_prompt, temperature=self.temperature, max_tokens=self.max_tokens, groq_api_key=self.api_keys.get("groq_api_key"))
            else:
                response = ollama.generate(
                    model=self.model,
                    prompt=agent_definition_prompt,
                    options={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                )
                response = response['response']
            agent_definitions = self.parse_json_response(response)
            if not agent_definitions:
                raise ValueError("Failed to generate valid agent definitions")
        except Exception as e:
            logging.error(f"Error generating agent definitions: {str(e)}")
            agent_definitions = self.fallback_agent_definitions()

        self.search_agents = {}
        for agent_data in agent_definitions.get("agents", []):
            agent_name = agent_data.get("name")
            role = agent_data.get("role")
            library = agent_data.get("library")
            prompt = agent_data.get("prompt")

            if agent_name and role and library and prompt:
                search_function = search_libraries.get(library)
                if search_function:
                    api_key = self.api_keys.get(f"{library}_api_key") or self.api_keys.get("serpapi_api_key")
                    cse_id = self.api_keys.get("google_cse_id") if library == "google" else None
                    if api_key or library == "duckduckgo":
                        self.search_agents[agent_name] = SearchAgent(
                            name=agent_name,
                            model=agent_model,
                            search_function=search_function,
                            api_key=api_key,
                            cse_id=cse_id,
                            prompt=prompt,
                            role=role
                        )
                    else:
                        logging.warning(f"Skipping agent {agent_name} due to missing API key for {library}")
                else:
                    logging.warning(f"Invalid search library: {library} for agent {agent_name}")
            else:
                logging.warning(f"Incomplete agent definition: {agent_data}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Attempts to parse the JSON response, handling potential errors.

        Args:
            response: The string response from the LLM.

        Returns:
            A dictionary containing the parsed JSON, or None if parsing fails.
        """
        try:
            # Remove any leading/trailing whitespace and newlines
            response = response.strip()
            # If the response is wrapped in ```, remove them
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            # Parse the JSON
            return json.loads(response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Problematic JSON: {response}")
            return None

    def fallback_agent_definitions(self) -> Dict[str, Any]:
        """
        Provides a fallback set of agent definitions if the LLM fails to generate valid JSON.

        Returns:
            A dictionary containing predefined agent definitions.
        """
        return {
            "agents": [
                {
                    "name": "General Researcher",
                    "role": "Background Research",
                    "library": "duckduckgo",
                    "prompt": "Conduct a general search on the topic and provide an overview of key information."
                },
                {
                    "name": "Detailed Analyst",
                    "role": "In-depth Analysis",
                    "library": "google",
                    "prompt": "Perform a detailed analysis of the topic, focusing on recent developments and expert opinions."
                },
                {
                    "name": "Fact Checker",
                    "role": "Verification",
                    "library": "bing",
                    "prompt": "Verify key claims and provide factual information from reputable sources."
                }
            ]
        }

    def run_research(self, user_request: str, report_length: str = "medium", agent_model: str = None, word_count_target: int = 1000) -> Tuple[str, List[str], List[Dict]]:
        self.create_search_agents(user_request, agent_model or self.model)

        agent_outputs = []
        all_search_results = {}
        citation_counter = 1
        for agent_name, agent in self.search_agents.items():
            logging.info(f"{agent_name} is working...")
            search_results = agent.search(user_request)
            for result in search_results:
                all_search_results[citation_counter] = result
                citation_counter += 1
            summary = agent.generate_summary(user_request, search_results)
            agent_outputs.append({"agent": agent_name, "summary": summary, "results": search_results})
            logging.info(f"{agent_name} summary: {summary}")
            yield f"{agent_name} Report", summary

        formatted_summaries = "\n\n".join([f"**{output['agent']} Summary:**\n{output['summary']}" for output in agent_outputs])
        final_report_prompt = f"""You are a research manager tasked with writing a {report_length} comprehensive report on: {user_request}

        Your team of specialized research agents has analyzed the topic and provided the following summaries:
        {formatted_summaries}

        Based on these summaries, generate a {report_length} report that answers the user's request.
        Aim for approximately {word_count_target} words in your report.
        Ensure the report is well-structured, informative, and includes citations in square brackets (e.g., [1], [2]).
        Include the following sections in your report:

        1. Introduction: Briefly introduce the topic and the purpose of the report.
        2. Background: Provide some background information on the topic.
        3. Analysis: Analyze the information from the agent summaries, drawing connections and insights.
        4. Conclusion: Summarize the key findings and provide a concluding statement.
        5. References: List the cited sources with their corresponding URLs.
        """
        logging.info("Search Manager is generating the final report...")
        try:
            if self.model in OPENAI_MODELS:
                final_report = call_openai_api(self.model, [{"role": "user", "content": final_report_prompt}], temperature=self.temperature, max_tokens=self.max_tokens, openai_api_key=self.api_keys.get("openai_api_key"))
            elif self.model in GROQ_MODELS:
                final_report = call_groq_api(self.model, final_report_prompt, temperature=self.temperature, max_tokens=self.max_tokens, groq_api_key=self.api_keys.get("groq_api_key"))
            else:
                response = ollama.generate(
                    model=self.model,
                    prompt=final_report_prompt,
                    options={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                )
                final_report = response['response']
        except Exception as e:
            logging.error(f"Error generating final report: {str(e)}")
            final_report = f"Error generating final report: {str(e)}"

        references = self.extract_references(final_report, all_search_results)
        
        yield "Final Report", final_report
        yield "References", references

        return final_report, references, agent_outputs

    def extract_references(self, report: str, all_search_results: Dict[int, Dict]) -> List[str]:
        """Extracts references from the generated report."""
        citation_pattern = r"\[(\d+)\]"
        citations = re.findall(citation_pattern, report)
        references = []
        for citation in citations:
            try:
                index = int(citation)
                result = all_search_results[index]
                references.append(f"[{citation}] {result['title']} - {result['url']}")
            except (ValueError, KeyError):
                references.append(f"Invalid citation: [{citation}]")
        return references