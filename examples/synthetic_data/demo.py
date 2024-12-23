from agentarium.agent import Agent
from agentarium.CheckpointManager import CheckpointManager


if __name__ == '__main__':

    checkpoint = CheckpointManager("demo")

    alice = Agent.create_agent()
    bob = Agent.create_agent()
    checkpoint.update(step="initialization")

    alice.talk_to(bob, "What a beautiful day!")
    checkpoint.update(step="interaction_1")

    checkpoint.save()
