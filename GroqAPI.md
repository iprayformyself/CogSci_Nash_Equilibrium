# Groq API Configuration

The LLM-based experimental conditions used the **Groq API** for remote inference.

The simulation was connected to the API through Python using authenticated HTTP requests. The API key was stored as an environment variable (`GROQ_API_KEY`) to avoid exposing credentials directly in the source code.

The experiment used the following model configuration:

| Parameter | Value |
| --- | --- |
| Provider | Groq API |
| Model | `llama-3.3-70b-versatile` |
| API Type | Chat completion API |
| Response Format | Structured JSON output |
| Integration Language | Python |

During each decision step, the simulation generated a structured prompt containing:

- the current game state,
- previous rounds,
- contribution history,
- and agent-specific instructions.

The prompt was then sent to the Groq API, and the returned response was parsed into structured fields such as:

- contribution decision,
- predicted contributions of other agents,
- confidence score,
- and reasoning summary.

The same LLM model was used across all LLM experimental conditions to ensure that differences in performance were caused by prompting structure rather than differences between models.
