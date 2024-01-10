import logging
from importlib.metadata import version
from typing import Any, Callable, Dict, List, Sequence, Type

import openai
from packaging.version import parse
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.types import ChatMessage

try:
    import konko

except ImportError:
    raise ValueError(
        "Could not import konko python package. "
        "Please install it with `pip install konko`."
    )

DEFAULT_KONKO_API_TYPE = "open_ai"
DEFAULT_KONKO_API_BASE = "https://api.konko.ai/v1"
DEFAULT_KONKO_API_VERSION = ""
MISSING_API_KEY_ERROR_MESSAGE = """No Konko API key found for LLM.
E.g. to use konko Please set the KONKO_API_KEY environment variable or \
konko.api_key prior to initialization.
API keys can be found or created at \
https://www.konko.ai/
"""

logger = logging.getLogger(__name__)


def is_openai_v1() -> bool:
    try:
        _version = parse(version("openai"))
        major_version = _version.major
    except AttributeError:
        # Handle the case where version or major attribute is not present
        return False
    return bool(major_version >= 1)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.APITimeoutError)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.APIStatusError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(is_chat_model: bool, max_retries: int, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        client = get_completion_endpoint(is_chat_model)
        return client.create(**kwargs)

    return _completion_with_retry(**kwargs)


def create_model_context_length_dict() -> dict:
    """
    Create a dictionary mapping Konko model names to their max context length.

    Returns:
    - dict: A dictionary where keys are model names and values are their max context length.
    """
    model_context_dict = {}

    if is_openai_v1():
        models = konko.models.list().data
        for model in models:
            model_name = model.name
            max_context_length = model.max_context_length
            model_context_dict[model_name] = max_context_length
    else:
        models = konko.Model.list().data
        for model in models:
            model_name = model["name"]
            max_context_length = model["max_context_length"]
            model_context_dict[model_name] = max_context_length

    return model_context_dict


ALL_AVAILABLE_MODELS = create_model_context_length_dict()


def konko_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = konko.modelname_to_contextsize(model_name)
    """
    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Konko model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_chat_model(model_id: str) -> bool:
    """
    Check if the specified model is a chat model.

    Args:
    - model_id (str): The ID of the model to check.

    Returns:
    - bool: True if the model is a chat model, False otherwise.

    Raises:
    - ValueError: If the model_id is not found in the list of models.
    """
    # Get the list of models based on the API version
    models = konko.models.list().data if is_openai_v1() else konko.Model.list().data

    # Use a generator expression to find the model by ID
    model = next(
        (m for m in models if (m.id if is_openai_v1() else m["id"]) == model_id), None
    )

    if model is None:
        raise ValueError(f"Model with ID {model_id} not found.")

    # Check if the model is a chat model
    return model.is_chat if is_openai_v1() else model["is_chat"]


def get_completion_endpoint(is_chat_model: bool) -> Any:
    """
    Get the appropriate completion endpoint based on the model type and API version.

    Args:
    - is_chat_model (bool): A flag indicating whether the model is a chat model.

    Returns:
    - The appropriate completion endpoint based on the model type and API version.

    Raises:
    - NotImplementedError: If the combination of is_chat_model and API version is not supported.
    """
    # For OpenAI version 1
    if is_openai_v1():
        return konko.chat.completions if is_chat_model else konko.completions

    # For other versions
    if not is_openai_v1():
        return konko.ChatCompletion if is_chat_model else konko.Completion

    # Raise error if the combination of is_chat_model and API version is not covered
    raise NotImplementedError(
        "The combination of model type and API version is not supported."
    )


def to_openai_message_dict(message: ChatMessage) -> dict:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role,
        "content": message.content,
    }
    message_dict.update(message.additional_kwargs)

    return message_dict


def to_openai_message_dicts(messages: Sequence[ChatMessage]) -> List[dict]:
    """Convert generic messages to OpenAI message dicts."""
    return [to_openai_message_dict(message) for message in messages]


def from_openai_message_dict(message_dict: Any) -> ChatMessage:
    """Convert openai message dict to generic message."""
    if is_openai_v1():
        role = message_dict.role
        content = message_dict.content
    else:
        role = message_dict["role"]
        content = message_dict.get("content", None)
    additional_kwargs = {
        attr: getattr(message_dict, attr)
        for attr in dir(message_dict)
        if not attr.startswith("_") and attr not in ["role", "content"]
    }
    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def from_openai_message_dicts(message_dicts: Sequence[dict]) -> List[ChatMessage]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message_dict(message_dict) for message_dict in message_dicts]


def to_openai_function(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI function."""
    schema = pydantic_class.schema()
    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": pydantic_class.schema(),
    }


async def acompletion_with_retry(
    is_chat_model: bool, max_retries: int, **kwargs: Any
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        if is_chat_model:
            if is_openai_v1():
                return await konko.AsyncKonko().chat.completions.create(**kwargs)
            else:
                return await konko.ChatCompletion.acreate(**kwargs)
        else:
            if is_openai_v1():
                return await konko.AsyncKonko().completions.create(**kwargs)
            else:
                return await konko.Completion.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)
