# Forge Human Rating Study — Instructions

Thank you for participating in the Forge human evaluation study. The purpose of this study is to collect human judgments of AI agent execution traces so we can measure how well Forge's automated metrics agree with human raters.

You will evaluate **10 AI agent execution traces** and rate each one on **5 dimensions**. Each trace shows the task, the agent's reasoning and tool-use steps, and its final answer. Rate the agent's *process and output* — not how you would have solved the task yourself.

> You will **not** see Forge's automated scores while rating. This is intentional: it keeps your judgment independent and avoids anchoring bias.

## Rating dimensions

### Question 1 — Task Completion
*Did the agent accomplish the task?*

- **0** — Agent completely failed or produced a wrong answer. *Example: asked for the capital of Japan, the agent answers "Beijing".*
- **1** — Agent partially completed the task. *Example: asked for a country's capital and its population, the agent gives the correct capital but no population.*
- **2** — Agent fully and correctly completed the task. *Example: asked for the capital of Japan, the agent answers "Tokyo" with correct supporting detail.*

### Question 2 — Reasoning Quality
*Are the reasoning steps coherent and on-topic?* **Evaluate the reasoning steps shown, not the final answer.**

- **1** — Reasoning steps are completely off-topic or contradictory.
- **2** — Reasoning is mostly off-topic with some relevant steps.
- **3** — Reasoning is somewhat relevant but with gaps.
- **4** — Reasoning is mostly coherent and relevant.
- **5** — Reasoning steps follow logically and stay on topic throughout.

### Question 3 — Tool Usage
*Did the agent use the right tools with the right inputs?*

- **1** — Agent used wrong tools or called tools with completely incorrect inputs.
- **2** — Agent used some correct tools but with suboptimal inputs or unnecessary calls.
- **3** — Agent used appropriate tools with correct inputs.

> Note: if the task required no tools and the agent used none, rate as **3**.

### Question 4 — Hallucination
*Does the final answer contain unsupported factual claims?*

- **0** — The final answer contains at least one factual claim that is incorrect or unsupported by the information retrieved.
- **1** — The final answer contains no identifiable incorrect factual claims.

> Note: if the agent said "I don't know" or produced no answer, rate as **1** (no hallucination detected).

### Question 5 — Efficiency
*Did the agent reach the answer directly?*

- **1** — Agent took many unnecessary steps to reach the answer.
- **2** — Agent took somewhat more steps than needed but reached the answer.
- **3** — Agent reached the answer in a minimal and direct way.

## How to submit

Record your ratings in the Google Form / spreadsheet provided by the study coordinator. Rate every trajectory before submitting. If a trajectory is unclear, provide your best rating and add a note explaining the uncertainty.
