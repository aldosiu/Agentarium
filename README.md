# Agentarium

A Python framework for managing and orchestrating AI agents.

## Installation

```bash
pip install agentarium
```

## Usage

```python
from agentarium import Agent, Environment, AgentInteractionManager

# Create an environment
env = Environment()

# Create agents
agent1 = Agent(name="agent1")
agent2 = Agent(name="agent2")

# Create interaction manager
manager = AgentInteractionManager()

# Add agents to the environment
env.add_agent(agent1)
env.add_agent(agent2)

# Start interactions
manager.start_interaction(env)
```

## Features

- Agent management and orchestration
- Flexible environment configuration
- Interaction management between agents

## License

This project is licensed under the MIT License - see the LICENSE file for details. 