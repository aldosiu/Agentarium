from __future__ import annotations

import re
import json
import aisuite as ai

from typing import List
from faker import Faker
from .Interaction import Interaction
from .AgentInteractionManager import AgentInteractionManager
from .Config import Config
from .utils import cache_w_checkpoint_manager


faker = Faker()
config = Config()
llm_client = ai.Client({**config.aisuite})


class Agent:
    """
    A class representing an autonomous agent in the environment.

    The Agent class is the core component of the Agentarium system, representing
    an individual entity capable of:
    - Maintaining its own identity and characteristics
    - Interacting with other agents through messages
    - Making autonomous decisions based on its state and interactions
    - Generating responses using natural language models

    Each agent has a unique identifier and a set of characteristics that define
    its personality and behavior. These characteristics can be either provided
    during initialization or generated automatically.

    Attributes:
        agent_id (str): Unique identifier for the agent
        agent_informations (dict): Dictionary containing agent characteristics
            including name, age, gender, occupation, location, and bio
        _interaction_manager (AgentInteractionManager): Singleton instance managing
            all agent interactions
    """

    _interaction_manager = AgentInteractionManager()
    _allow_init = False

    _default_generate_agent_prompt = """You're goal is to generate the bio of a fictional person.
Make this bio as realistic and as detailed as possible.
You may be given information about the person to generate a bio for, if so, use that information to generate a bio.
If you are not given any information about the person, generate a bio for a random person.

You must generate the bio in the following format:
Bio: [Bio of the person]
"""

    _default_act_prompt = """
Informations about yourself:
{agent_informations}

Your interactions:
{interactions}

Given the above information, think about what you should do next.

The following are the possible actions you can take:
[THINK][CONTENT]: Think about something. (i.e [THINK][It's such a beautiful day!])
[TALK][AGENT_ID][CONTENT]: Talk to the agent with the given ID. Note that you must the agent ID, not the agent name. (i.e [TALK][123][Hello!])

Write in the following format:
<THINK>
[YOUR_THOUGHTS]
</THINK>

<ACTION>
[YOUR_NEXT_ACTION: One of the following actions: THINK, TALK]
</ACTION>
"""

    def __init__(self, **kwargs):
        """
        Initialize an agent with given or generated characteristics.
        This method should not be called directly - use create_agent() instead.
        """

        if not Agent._allow_init:
            raise RuntimeError("Agent instances should be created using Agent.create_agent()")

        self.agent_id = kwargs.get("agent_id", faker.uuid4())
        self.agent_informations: dict = kwargs or {}

        if "gender" not in kwargs:
            self.agent_informations["gender"] = faker.random_element(elements=["male", "female"])

        if "name" not in kwargs:
            self.agent_informations["name"] = getattr(faker, f"name_{self.agent_informations['gender']}")()

        if "age" not in kwargs:
            self.agent_informations["age"] = faker.random_int(18, 80)

        if "occupation" not in kwargs:
            self.agent_informations["occupation"] = faker.job()

        if "location" not in kwargs:
            self.agent_informations["location"] = faker.city()

        if "bio" not in kwargs:
            self.agent_informations["bio"] = Agent._generate_agent_bio(self.agent_informations)

        self._interaction_manager.register_agent(self)

    @staticmethod
    @cache_w_checkpoint_manager
    def create_agent(**kwargs) -> Agent:
        """
        Initialize an agent with given or generated characteristics.

        Creates a new agent instance with a unique identifier and a set of
        characteristics. If specific characteristics are not provided, they
        are automatically generated to create a complete and realistic agent
        profile.

        The following characteristics are handled:
        - Gender (male/female)
        - Name (appropriate for the gender)
        - Age (between 18 and 80)
        - Occupation (randomly selected job)
        - Location (randomly selected city)
        - Bio (generated based on other characteristics)

        Args:
            **kwargs: Dictionary of agent characteristics to use instead of
                generating them. Any characteristic not provided will be
                automatically generated.
        """
        try:
            Agent._allow_init = True
            return Agent(**kwargs)
        finally:
            Agent._allow_init = False

    @property
    def name(self) -> str:
        """
        Get the agent's name.

        Returns:
            str: The name of the agent.
        """
        return self.agent_informations["name"]

    @staticmethod
    def _generate_prompt_to_generate_bio(**kwargs) -> str:
        """
        Generate a prompt for creating an agent's biography.

        Creates a prompt that will be used by the language model to generate
        a realistic biography for the agent. If characteristics are provided,
        they are incorporated into the prompt to ensure the generated bio
        is consistent with the agent's existing traits.

        Args:
            **kwargs: Dictionary of agent characteristics to incorporate
                into the biography generation prompt.

        Returns:
            str: A formatted prompt string for biography generation.
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if not kwargs:
            return Agent._default_generate_agent_prompt

        prompt = Agent._default_generate_agent_prompt
        prompt += "\nInformation about the person to generate a bio for:"

        for key, value in kwargs.items():
            prompt += f"\n{key}: {value}"

        return prompt

    @staticmethod
    def _generate_agent_bio(agent_informations: dict) -> str:
        """
        Generate a biography for an agent using a language model.

        Uses the OpenAI API to generate a realistic and detailed biography
        based on the agent's characteristics. The biography is generated
        to be consistent with any existing information about the agent.

        Args:
            agent_informations (dict): Dictionary of agent characteristics
                to use in generating the biography.

        Returns:
            str: A generated biography for the agent.
        """
        prompt = Agent._generate_prompt_to_generate_bio(**agent_informations)

        response = llm_client.chat.completions.create(
            model=f"{config.llm_provider}:{config.llm_model}",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )

        return response.choices[0].message.content

    @cache_w_checkpoint_manager
    def act(self) -> str:
        """
        Generate and execute the agent's next action.

        Uses the language model to determine and perform the agent's next
        action based on their characteristics and interaction history.
        The agent can either think to themselves or talk to another agent.

        Returns:
            str: The complete response from the language model, including
                the agent's thoughts and chosen action.

        Raises:
            RuntimeError: If the action format is invalid or if the target
                agent for a TALK action is not found.
        """

        prompt = Agent._default_act_prompt.format(
            agent_informations=self.agent_informations,
            interactions=self.get_interactions(),
        )

        response = llm_client.chat.completions.create(
            model=f"{config.llm_provider}:{config.llm_model}",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        actions = re.search(r"<ACTION>(.*?)</ACTION>", response.choices[0].message.content, re.DOTALL).group(1).strip().split("]")
        actions = [action.replace("[", "").replace("]", "").strip() for action in actions]

        if actions[0] == "TALK":
            if len(actions) < 3:
                raise RuntimeError(f"Received a TALK action with less than 3 arguments: {actions}")

            if (receiver := self._interaction_manager.get_agent(actions[1])) is None:
                raise RuntimeError(f"Received a TALK action with an invalid agent ID: {actions[1]}")

            self.talk_to(receiver, actions[2])

        elif actions[0] == "THINK":
            # NOTE: This is a self-interaction, but maybe we should add a way to handle self-thinking.
            self._interaction_manager.record_interaction(self, self, actions[1])

        else:
            raise RuntimeError(f"Invalid action: {actions[0]}")

        return response.choices[0].message.content

    def dump(self) -> dict:
        """
        Dump the agent's state to a dictionary.
        """

        return {
            "agent_id": self.agent_id,
            "agent_informations": self.agent_informations,
            "interactions": [interaction.dump() for interaction in self._interaction_manager._agent_private_interactions[self.agent_id]],
        }

    def __str__(self) -> str:
        """
        Get a string representation of the agent.

        Returns:
            str: A formatted string containing all the agent's characteristics.
        """
        return "\n".join([f"{key.capitalize()}: {value}" for key, value in self.agent_informations.items()])

    def __repr__(self) -> str:
        """
        Get a string representation of the agent, same as __str__.

        Returns:
            str: A formatted string containing all the agent's characteristics.
        """
        return Agent.__str__(self)

    def get_interactions(self) -> List[Interaction]:
        """
        Retrieve all interactions involving this agent.

        Returns:
            List[Interaction]: A list of all interactions where this agent
                was either the sender or receiver.
        """
        return self._interaction_manager.get_agent_interactions(self)

    @cache_w_checkpoint_manager
    def talk_to(self, receiver: Agent, message: str) -> None:
        """
        Send a message to another agent.

        Records an interaction where this agent sends a message to another agent.

        Args:
            receiver (Agent): The agent to send the message to.
            message (str): The content of the message to send.
        """
        self._interaction_manager.record_interaction(self, receiver, message)


if __name__ == "__main__":

    interaction = Interaction(
        sender=Agent(name="Alice", bio="Alice is a software engineer."),
        receiver=Agent(name="Bob", bio="Bob is a data scientist."),
        message="Hello Bob! I heard you're working on some interesting data science projects."
    )

    print(interaction)
