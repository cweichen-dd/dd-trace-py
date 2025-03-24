from hashlib import sha1
import json
from re import match
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ddtrace import config
from ddtrace.ext import SpanTypes
from ddtrace.internal.logger import get_logger
from ddtrace.llmobs._constants import GEMINI_APM_SPAN_NAME
from ddtrace.llmobs._constants import DEFAULT_PROMPT_NAME
from ddtrace.llmobs._constants import INTERNAL_CONTEXT_VARIABLE_KEYS
from ddtrace.llmobs._constants import INTERNAL_QUERY_VARIABLE_KEYS
from ddtrace.llmobs._constants import IS_EVALUATION_SPAN
from ddtrace.llmobs._constants import LANGCHAIN_APM_SPAN_NAME
from ddtrace.llmobs._constants import ML_APP
from ddtrace.llmobs._constants import NAME
from ddtrace.llmobs._constants import OPENAI_APM_SPAN_NAME
from ddtrace.llmobs._constants import SESSION_ID
from ddtrace.llmobs._constants import VERTEXAI_APM_SPAN_NAME
from ddtrace.llmobs.utils import Message
from ddtrace.llmobs.utils import Prompt
from ddtrace.trace import Span


log = get_logger(__name__)


def validate_prompt(prompt: Prompt, ml_app:str="", strict_validation=True) -> Dict[str, Union[str, Dict[str, Any], List[str], List[Dict[str, str]], List[Message]]]:
    validated_prompt = {}

    if not isinstance(prompt, dict):
        raise TypeError("Prompt must be a dictionary")

    name = prompt.get("name")
    variables = prompt.get("variables")
    template = prompt.get("template")
    chat_template = prompt.get("chat_template")
    version = prompt.get("version")
    prompt_id = prompt.get("id")
    ctx_variable_keys = prompt.get("rag_context_variables")
    rag_query_variable_keys = prompt.get("rag_query_variables")

    if strict_validation:
        if template is None and chat_template is None:
            raise ValueError("Prompt must have a template or chat template.")

    if name is not None:
        if not isinstance(name, str):
            raise TypeError("Prompt name must be a string.")
        validated_prompt["name"] = name
    elif prompt_id is not None:
        validated_prompt["name"] = prompt_id
    else:
        validated_prompt["name"] = DEFAULT_PROMPT_NAME

    if prompt_id is None:
        if strict_validation:
            raise ValueError("Prompt ID is not provided.")
        log.warning("Prompt ID is not provided. The prompt ID will be generated based on the prompt name.")
        validated_prompt["id"] = f"{ml_app}-{name or DEFAULT_PROMPT_NAME}"
    elif not isinstance(prompt_id, str):
        raise TypeError("Prompt ID must be a string.")
    else:
        validated_prompt["id"] = prompt_id

    if version is not None:
        semver_regex = (
            r"^(?P<major>0|[1-9]\d*)\."
            r"(?P<minor>0|[1-9]\d*)\."
            r"(?P<patch>0|[1-9]\d*)"
            r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-]"
            r"[0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-]"
            r"[0-9a-zA-Z-]*))*))?"
            r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+"
            r"(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        if not isinstance(version, str):
            raise TypeError("Prompt version must be a string.")
        if not bool(match(semver_regex, version)):
            if strict_validation:
                raise ValueError("Prompt version must be semver compatible.")
            else :
                log.warning(
                    "Prompt version must be semver compatible. Please check https://semver.org/ for more information."
                )
        # Add minor and patch version if not present
        version_parts = (version.split(".") + ["0", "0"])[:3]
        version = ".".join(version_parts)
        validated_prompt["version"] = version
    else:
        validated_prompt["version"] = "1.0.0"

    if variables is not None:
        if not isinstance(variables, dict):
            raise TypeError("Prompt variables must be a dictionary.")
        if not any(isinstance(k, str) for k in variables):
            raise TypeError("Prompt variable keys must be strings.")
        # check if values are json serializable
        if not all(isinstance(v, (str, int, float, bool, list, dict)) for v in variables.values()):
            raise TypeError("Prompt variable values must be JSON serializable.")
        validated_prompt["variables"] = variables

    if strict_validation and (template is None and chat_template is None):
        raise ValueError("Prompt must have a template or chat template.")
    elif template:
        if not isinstance(template, str):
            raise TypeError("Prompt template must be a string")
        validated_prompt["template"] = template
    elif chat_template:
        validated_chat_template = []
        # accept a single message as a chat template
        if isinstance(chat_template, Message):
            validated_chat_template.append(chat_template)
        elif isinstance(chat_template, list):
            for message in chat_template:
                # accept a list of messages as a chat template
                if isinstance(message, Message):
                    validated_chat_template.append(message)
                # check if chat_template is a list of tuples of 2 strings and transform into messages structure
                elif isinstance(message, tuple) and len(message) == 2 and all(isinstance(t, str) for t in message):
                    validated_chat_template.append({"role": message[0], "content": message[1]})
                else:
                    raise TypeError("Prompt chat_template entry must be a Message or a 2-tuple (role,content).")
        else :
            raise TypeError("Prompt chat_template must be a list of Message objects or a list of 2-tuples (role,content).")
        validated_prompt["chat_template"] = validated_chat_template

    if ctx_variable_keys is not None:
        if not isinstance(ctx_variable_keys, list):
            raise TypeError("Prompt field `context_variable_keys` must be a list of strings.")
        if not all(isinstance(k, str) for k in ctx_variable_keys):
            raise TypeError("Prompt field `context_variable_keys` must be a list of strings.")
        validated_prompt[INTERNAL_CONTEXT_VARIABLE_KEYS] = ctx_variable_keys
    else:
        validated_prompt[INTERNAL_CONTEXT_VARIABLE_KEYS] = ["context"]

    if rag_query_variable_keys is not None:
        if not isinstance(rag_query_variable_keys, list):
            raise TypeError("Prompt field `rag_query_variables` must be a list of strings.")
        if not all(isinstance(k, str) for k in rag_query_variable_keys):
            raise TypeError("Prompt field `rag_query_variables` must be a list of strings.")
        validated_prompt[INTERNAL_QUERY_VARIABLE_KEYS] = rag_query_variable_keys
    else:
        validated_prompt[INTERNAL_QUERY_VARIABLE_KEYS] = ["question"]

    # Compute prompt instance id
    validated_prompt["prompt_instance_id"] = _get_prompt_instance_id(validated_prompt, ml_app)

    return validated_prompt


def _get_prompt_instance_id(validated_prompt: dict, ml_app: str) -> str:
    name = validated_prompt.get("name")
    variables = validated_prompt.get("variables")
    template = validated_prompt.get("template")
    chat_template = validated_prompt.get("chat_template")
    version = validated_prompt.get("version")
    prompt_id = validated_prompt.get("id")
    ctx_variable_keys = validated_prompt.get("rag_context_variables")
    rag_query_variable_keys = validated_prompt.get("rag_query_variables")

    instance_id_str = (f"[{ml_app}]{prompt_id}"
                       f"{name}{prompt_id}{version}{template}{chat_template}{variables}"
                       f"{ctx_variable_keys}{rag_query_variable_keys}")

    prompt_instance_id = sha1(instance_id_str.encode()).hexdigest()

    return prompt_instance_id


class LinkTracker:
    def __init__(self, object_span_links=None):
        self._object_span_links = object_span_links or {}

    def get_object_id(self, obj):
        return f"{type(obj).__name__}_{id(obj)}"

    def add_span_links_to_object(self, obj, span_links):
        obj_id = self.get_object_id(obj)
        if obj_id not in self._object_span_links:
            self._object_span_links[obj_id] = []
        self._object_span_links[obj_id] += span_links

    def get_span_links_from_object(self, obj):
        return self._object_span_links.get(self.get_object_id(obj), [])


class AnnotationContext:
    def __init__(self, _register_annotator, _deregister_annotator):
        self._register_annotator = _register_annotator
        self._deregister_annotator = _deregister_annotator

    def __enter__(self):
        self._register_annotator()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._deregister_annotator()

    async def __aenter__(self):
        self._register_annotator()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._deregister_annotator()


def _get_attr(o: object, attr: str, default: object):
    # Convenience method to get an attribute from an object or dict
    if isinstance(o, dict):
        return o.get(attr, default)
    return getattr(o, attr, default)


def _get_nearest_llmobs_ancestor(span: Span) -> Optional[Span]:
    """Return the nearest LLMObs-type ancestor span of a given span."""
    parent = span._parent
    while parent:
        if parent.span_type == SpanTypes.LLM:
            return parent
        parent = parent._parent
    return None


def _get_span_name(span: Span) -> str:
    if span.name in (LANGCHAIN_APM_SPAN_NAME, GEMINI_APM_SPAN_NAME, VERTEXAI_APM_SPAN_NAME) and span.resource != "":
        return span.resource
    elif span.name == OPENAI_APM_SPAN_NAME and span.resource != "":
        client_name = span.get_tag("openai.request.client") or "OpenAI"
        return "{}.{}".format(client_name, span.resource)
    return span._get_ctx_item(NAME) or span.name


def _is_evaluation_span(span: Span) -> bool:
    """
    Return whether or not a span is an evaluation span by checking the span's
    nearest LLMObs span ancestor. Default to 'False'
    """
    is_evaluation_span = span._get_ctx_item(IS_EVALUATION_SPAN)
    if is_evaluation_span:
        return is_evaluation_span
    llmobs_parent = _get_nearest_llmobs_ancestor(span)
    while llmobs_parent:
        is_evaluation_span = llmobs_parent._get_ctx_item(IS_EVALUATION_SPAN)
        if is_evaluation_span:
            return is_evaluation_span
        llmobs_parent = _get_nearest_llmobs_ancestor(llmobs_parent)
    return False


def _get_ml_app(span: Span) -> str:
    """
    Return the ML app name for a given span, by checking the span's nearest LLMObs span ancestor.
    Default to the global config LLMObs ML app name otherwise.
    """
    ml_app = span._get_ctx_item(ML_APP)
    if ml_app:
        return ml_app
    llmobs_parent = _get_nearest_llmobs_ancestor(span)
    while llmobs_parent:
        ml_app = llmobs_parent._get_ctx_item(ML_APP)
        if ml_app is not None:
            return ml_app
        llmobs_parent = _get_nearest_llmobs_ancestor(llmobs_parent)
    return ml_app or config._llmobs_ml_app or "unknown-ml-app"


def _get_session_id(span: Span) -> Optional[str]:
    """Return the session ID for a given span, by checking the span's nearest LLMObs span ancestor."""
    session_id = span._get_ctx_item(SESSION_ID)
    if session_id:
        return session_id
    llmobs_parent = _get_nearest_llmobs_ancestor(span)
    while llmobs_parent:
        session_id = llmobs_parent._get_ctx_item(SESSION_ID)
        if session_id is not None:
            return session_id
        llmobs_parent = _get_nearest_llmobs_ancestor(llmobs_parent)
    return session_id


def _unserializable_default_repr(obj):
    try:
        return str(obj)
    except Exception:
        log.warning("I/O object is neither JSON serializable nor string-able. Defaulting to placeholder value instead.")
        return "[Unserializable object: {}]".format(repr(obj))


def safe_json(obj, ensure_ascii=True):
    if isinstance(obj, str):
        return obj
    try:
        # If object is a Pydantic model, convert to JSON serializable dict first using model_dump()
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            obj = obj.model_dump()
        return json.dumps(obj, ensure_ascii=ensure_ascii, skipkeys=True, default=_unserializable_default_repr)
    except Exception:
        log.error("Failed to serialize object to JSON.", exc_info=True)
