from altk.pre_tool.refraction.src import refract
from tests.utils.refraction.tools.sample_tool_specs_untyped import (
    tools_internal,
    tool_calls_internal,
    tool_calls,
    tools,
)


class TestUntypedInputs:
    def test_success(self) -> None:
        result = refract(sequence=tool_calls, catalog=tools)
        assert result.report.determination is False

    def test_success_internal(self) -> None:
        result = refract(sequence=tool_calls_internal, catalog=tools_internal)
        assert result.report.determination is False
