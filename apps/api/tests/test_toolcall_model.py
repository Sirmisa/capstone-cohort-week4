"""
Tests for ToolCall and AgentResponse Pydantic models.

Covers the bug where GPT-4.1-mini returns tool calls with "parameters"
instead of "arguments", causing Pydantic validation to fail.
See: BUG_REPORT_TOOLCALL_PARAMETERS_VS_ARGUMENTS.md
"""

import pytest
from pydantic import ValidationError
from api.agents.agents import ToolCall, AgentResponse


# ---------------------------------------------------------------------------
# ToolCall model
# ---------------------------------------------------------------------------

class TestToolCallWithArguments:
    """Standard case: LLM returns 'arguments' as expected."""

    def test_basic_tool_call(self):
        tc = ToolCall(name="get_formatted_items_context", arguments={"query": "earphones", "top_k": 5})
        assert tc.name == "get_formatted_items_context"
        assert tc.arguments == {"query": "earphones", "top_k": 5}

    def test_empty_arguments(self):
        tc = ToolCall(name="some_tool", arguments={})
        assert tc.arguments == {}

    def test_from_dict_with_arguments(self):
        data = {"name": "get_formatted_items_context", "arguments": {"query": "laptop bag"}}
        tc = ToolCall.model_validate(data)
        assert tc.arguments == {"query": "laptop bag"}


class TestToolCallWithParameters:
    """Bug scenario: LLM returns 'parameters' instead of 'arguments'."""

    def test_parameters_key_is_accepted(self):
        """This is the core bug fix test — parameters should be normalized to arguments."""
        data = {"name": "get_formatted_reviews_context", "parameters": {"query": "dinosaur headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}}
        tc = ToolCall.model_validate(data)
        assert tc.name == "get_formatted_reviews_context"
        assert tc.arguments == {"query": "dinosaur headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}

    def test_parameters_key_from_json(self):
        """Simulate instructor parsing the LLM JSON response."""
        json_str = '{"name": "get_formatted_reviews_context", "parameters": {"query": "headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}}'
        tc = ToolCall.model_validate_json(json_str)
        assert tc.arguments == {"query": "headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}

    def test_arguments_takes_precedence_over_parameters(self):
        """If both are present, arguments should win."""
        data = {
            "name": "some_tool",
            "arguments": {"key": "from_arguments"},
            "parameters": {"key": "from_parameters"},
        }
        tc = ToolCall.model_validate(data)
        assert tc.arguments == {"key": "from_arguments"}


class TestToolCallValidation:
    """Edge cases and validation."""

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"arguments": {"query": "test"}})

    def test_missing_both_arguments_and_parameters_raises(self):
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"name": "some_tool"})


# ---------------------------------------------------------------------------
# AgentResponse model (integration with ToolCall)
# ---------------------------------------------------------------------------

class TestAgentResponseWithToolCalls:
    """Ensure AgentResponse correctly validates nested ToolCall objects."""

    def test_response_with_arguments_tool_calls(self):
        data = {
            "answer": "Fetching reviews...",
            "references": [],
            "final_answer": False,
            "tool_calls": [
                {"name": "get_formatted_reviews_context", "arguments": {"query": "headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}}
            ],
        }
        resp = AgentResponse.model_validate(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].arguments["query"] == "headphones"

    def test_response_with_parameters_tool_calls(self):
        """The exact scenario from the Langsmith trace that caused the crash."""
        data = {
            "answer": "I will fetch the top reviews for the QearFun Dinosaur Headphones.",
            "references": [],
            "final_answer": False,
            "tool_calls": [
                {"name": "get_formatted_reviews_context", "parameters": {"query": "dinosaur headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}}
            ],
        }
        resp = AgentResponse.model_validate(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].arguments["item_list"] == ["B0B67M9C9P"]

    def test_response_with_no_tool_calls(self):
        data = {
            "answer": "Here are the details about the laptop bag...",
            "references": [{"id": "B07F9MFVKS", "description": "EZrelia Laptop Bag"}],
            "final_answer": True,
            "tool_calls": [],
        }
        resp = AgentResponse.model_validate(data)
        assert resp.final_answer is True
        assert resp.tool_calls == []

    def test_response_with_multiple_tool_calls_mixed_keys(self):
        """Some tool calls use arguments, some use parameters."""
        data = {
            "answer": "Looking up items...",
            "references": [],
            "final_answer": False,
            "tool_calls": [
                {"name": "get_formatted_items_context", "arguments": {"query": "earphones", "top_k": 5}},
                {"name": "get_formatted_reviews_context", "parameters": {"query": "headphones", "item_list": ["B0B67M9C9P"], "top_k": 10}},
            ],
        }
        resp = AgentResponse.model_validate(data)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].arguments["query"] == "earphones"
        assert resp.tool_calls[1].arguments["query"] == "headphones"
