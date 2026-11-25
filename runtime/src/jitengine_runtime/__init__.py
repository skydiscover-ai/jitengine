"""What every JITEngine engine serves the same way, whatever its model and Mac: the Anthropic Messages API
(anthropic_api) and OpenAI's Responses API (responses_api), each read into the chat-completions request the engine's
own server serves and written back from its reply; and Hugging Face model ids for `--model-path` (hub). It imports no
engine; of its dependencies only hub needs huggingface_hub, and imports it when a model id is fetched."""
