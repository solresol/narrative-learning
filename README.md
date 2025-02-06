# narrative-learning

What if a text-based explanation was the machine learning model?


## Explanation

Traditional machine learning follows a pattern: train a model, then generate explanations so humans can understand its decision-making process. These explanations are meant to be clear enough that a human could theoretically reproduce the model's decisions. But what if we flipped this paradigm?
With the emergence of Large Language Models (LLMs) that can effectively parse and act on natural language instructions, we can now explore an intriguing possibility: using human-readable explanations as the model itself. Instead of treating explanations as post-hoc justifications of a black-box model, we can iteratively refine natural language rules that directly drive the decision-making process.
This approach offers several potential advantages:

- **Inherent Interpretability**: The model's logic is explicitly encoded in human-readable form from the start, eliminating the need for separate explanation methods

- **Interactive Refinement**: We can leverage LLMs to improve the rules based on performance, while maintaining human-understandable language

- **Verification by Inspection**: Domain experts can directly review, validate, and suggest improvements to the decision-making criteria

- **Flexible Deployment**: The same rules can be interpreted by different LLMs, potentially allowing for trade-offs between accuracy and computational efficiency

### Some research questions



- **Model Complexity**: How sophisticated does an LLM need to be to effectively interpret and apply natural language rules?

- **Training Efficiency**: How does the performance of narrative learning compare to traditional classification techniques in terms of sample efficiency and convergence?

- **Rule Evolution**: What patterns emerge in how the natural language rules evolve through iterations of refinement?

- **Context Dependence**: How sensitive is performance to the amount of historical context or number of examples provided?

- **Generalization**: How well do narrative rules learned from one domain transfer to related problems?

- **Regression**: This code shows how to do classifiers. How would we do narrative learning *regressors*?


## Data prep

`python initialise_database.py`

Maybe do this, but this might be a bad idea after a few inferences. It's not needed
for any other step.

`sqlite3 titanic_medical.sqlite ".dump" > titanic_medical.sql`

Check to see if it works... this should produce a long list:

`./process_round.py --list --round 1`


## Operation

This will get things going.

`./process_round.py --list --round 1` 

I haven't implemented the actual training yet.
