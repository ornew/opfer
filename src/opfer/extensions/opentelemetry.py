import json
from typing import ContextManager, Mapping, Sequence

import opentelemetry.sdk.resources
import opentelemetry.sdk.trace
import opentelemetry.sdk.trace.export
import opentelemetry.trace

from opfer.tracing import (
    Attributes,
    ReadableSpan,
    SpanContext,
    SpanLink,
    SpanStatus,
    SpanStatusValue,
)
from opfer.types import JsonValue


def _as_otel_attribute_value(
    value: JsonValue,
) -> str | int | float | bool | list[str] | list[bool] | list[float] | list[str] | None:
    match value:
        case None:
            return None
        case str():
            return value
        case bool():
            return value
        case int():
            return value
        case float():
            return value
        case _:
            if isinstance(value, Sequence):
                return json.dumps(value)
            if isinstance(value, Mapping):
                return json.dumps(value)


def _as_otel_attributes(
    attributes: Attributes,
) -> Mapping[
    str,
    str
    | int
    | float
    | bool
    | Sequence[str]
    | Sequence[bool]
    | Sequence[float]
    | Sequence[str],
]:
    c = {key: _as_otel_attribute_value(value) for key, value in attributes.items()}
    return {key: value for key, value in c.items() if value is not None}


def _as_otel_links(link: SpanLink) -> opentelemetry.trace.Link:
    otel_span: opentelemetry.trace.Span = link.context.extras["otel_span"]
    otel_span_ctx = otel_span.get_span_context()
    return opentelemetry.trace.Link(
        context=opentelemetry.trace.SpanContext(
            trace_id=otel_span_ctx.trace_id,
            span_id=otel_span_ctx.span_id,
            is_remote=False,
        ),
        attributes={
            **(_as_otel_attributes(link.attributes) if link.attributes else {}),
            "opfer.trace_id": link.context.trace_id,
            "opfer.span_id": link.context.span_id,
        },
    )


class OtelSpanProcessor:
    def __init__(self, otel_tracer: opentelemetry.trace.Tracer):
        self.tracer = otel_tracer

    def on_start(
        self,
        span: ReadableSpan,
        parent_context: SpanContext | None = None,
    ) -> None:
        print(f"Starting span: {span.name}")
        otel_span = self.tracer.start_span(
            name=span.name,
            start_time=int(span.start_time.timestamp() * 1_000_000_000),
            attributes=_as_otel_attributes(span.attributes),
            links=[_as_otel_links(link) for link in span.links],
        )
        otel_span.set_attributes(
            {
                "opfer.trace_id": span.context.trace_id,
                "opfer.span_id": span.context.span_id,
            }
        )
        ctx = opentelemetry.trace.use_span(otel_span, end_on_exit=False)
        ctx.__enter__()
        span.context.extras["otel_span"] = otel_span
        span.context.extras["otel_ctx"] = ctx

    def on_end(self, span: ReadableSpan) -> None:
        print(f"Ending span: {span.name}")
        otel_span: opentelemetry.trace.Span = span.context.extras["otel_span"]
        otel_span.set_attributes(_as_otel_attributes(span.attributes))
        for link in span.links:
            c = _as_otel_links(link)
            otel_span.add_link(c.context, c.attributes)
        for event in span.events:
            otel_span.add_event(
                name=event.name,
                attributes=_as_otel_attributes(event.attributes)
                if event.attributes
                else None,
                timestamp=int(event.timestamp.timestamp() * 1_000_000_000),
            )
        match span.status:
            case SpanStatus(value=value, description=description):
                match value:
                    case SpanStatusValue.UNSET:
                        pass
                    case SpanStatusValue.OK:
                        otel_span.set_status(
                            opentelemetry.trace.StatusCode.OK, description
                        )
                    case SpanStatusValue.ERROR:
                        otel_span.set_status(
                            opentelemetry.trace.StatusCode.ERROR, description
                        )

        otel_span.end(
            end_time=int(span.end_time.timestamp() * 1_000_000_000)
            if span.end_time
            else None
        )
        otel_span_ctx: ContextManager[opentelemetry.trace.Span] = span.context.extras[
            "otel_ctx"
        ]
        otel_span_ctx.__exit__(None, None, None)

    def flush(self) -> None: ...

    def close(self) -> None: ...
