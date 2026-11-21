from collections.abc import AsyncIterator

import httpx
import pytest
from openai import APIConnectionError
from openai.types.responses import ResponseFailedEvent, ResponseIncompleteEvent

from agents import (
    Agent,
    ModelBehaviorError,
    ModelRetrySettings,
    ModelSettings,
    RunConfig,
    Runner,
    Usage,
    retry_policies,
)
from agents.testing import ModelCall, ModelStep, ScriptedModel

from .model_test_helpers import get_response_obj


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["response.failed", "response.incomplete"])
@pytest.mark.parametrize("include_usage", [True, False], ids=["usage", "no-usage"])
@pytest.mark.parametrize("retry_first_attempt", [True, False], ids=["retry", "no-retry"])
async def test_streamed_terminal_failure_accounts_for_retry_attempts(
    event_type: str, include_usage: bool, retry_first_attempt: bool
) -> None:
    async def stream(
        _call: ModelCall,
    ) -> AsyncIterator[ResponseFailedEvent | ResponseIncompleteEvent]:
        response = get_response_obj(
            [], usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        )
        if not include_usage:
            response.usage = None
        if event_type == "response.failed":
            yield ResponseFailedEvent(type="response.failed", response=response, sequence_number=0)
        else:
            yield ResponseIncompleteEvent(
                type="response.incomplete", response=response, sequence_number=0
            )

    model = ScriptedModel()
    if retry_first_attempt:
        model.enqueue(APIConnectionError(request=httpx.Request("POST", "https://example.invalid")))
    model.enqueue(ModelStep.stream(stream))
    agent = Agent(
        name="test",
        model=model,
        model_settings=ModelSettings(
            retry=ModelRetrySettings(
                max_retries=1,
                policy=retry_policies.network_error(),
                backoff={"initial_delay": 0},
            )
        ),
    )
    result = Runner.run_streamed(agent, "test", run_config=RunConfig(tracing_disabled=True))

    with pytest.raises(ModelBehaviorError, match=event_type):
        async for _ in result.stream_events():
            pass

    assert len(model.calls) == (2 if retry_first_attempt else 1)
    usage = result.context_wrapper.usage
    assert usage.requests == (2 if retry_first_attempt else int(include_usage))
    assert len(usage.request_usage_entries) == usage.requests
    assert usage.input_tokens == (10 if include_usage else 0)
    assert usage.output_tokens == (5 if include_usage else 0)
    assert usage.total_tokens == (15 if include_usage else 0)
    if retry_first_attempt:
        assert usage.request_usage_entries[0].total_tokens == 0
        assert usage.request_usage_entries[1].total_tokens == (15 if include_usage else 0)
