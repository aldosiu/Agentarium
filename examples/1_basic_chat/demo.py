from agentarium import Agent

# Create some agents
alice_agent = Agent.create_agent(name="Alice", occupation="Software Engineer")
bob_agent = Agent.create_agent(name="Bob", occupation="Data Scientist")

alice_agent.talk_to(bob_agent, "Hello Bob! I heard you're working on some interesting data science projects.")
bob_agent.talk_to(alice_agent, "Hi Alice! Yes, I'm currently working on a machine learning model for natural language processing.")

alice_agent.act() # Here it's the agent that decides to talk to Bob
bob_agent.act()

print("\n === Interactions ===\n")
print("Alice's interactions:")
print(alice_agent.get_interactions())

print("\nBob's interactions:")
print(bob_agent.get_interactions())
