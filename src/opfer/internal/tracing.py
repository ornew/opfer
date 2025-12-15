from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Protocol,
    Sequence,
)
from uuid import uuid4

from opfer.types import JsonValue

type AttributeValue = JsonValue
type Attributes = Mapping[str, AttributeValue]
type MutableAttributes = MutableMapping[str, AttributeValue]

type Timestamp = datetime


def now() -> Timestamp:
    return datetime.now(timezone.utc)


@dataclass
class SpanContext:
    trace_id: str
    span_id: str
    extras: MutableMapping[str, Any] = field(default_factory=dict)


class SpanStatusValue(str, Enum):
    UNSET = "UNSET"
    ERROR = "ERROR"
    OK = "OK"


@dataclass
class SpanStatus:
    value: SpanStatusValue = field(default=SpanStatusValue.UNSET)
    description: str | None = field(default=None)


@dataclass
class SpanEvent:
    name: str
    attributes: Attributes
    timestamp: Timestamp


@dataclass
class SpanLink:
    context: SpanContext
    attributes: Attributes | None


class ReadableSpan(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def context(self) -> SpanContext: ...

    @property
    def parent(self) -> SpanContext | None: ...

    @property
    def status(self) -> SpanStatus: ...

    @property
    def attributes(self) -> Attributes: ...

    @property
    def start_time(self) -> Timestamp: ...

    @property
    def end_time(self) -> Timestamp | None: ...

    @property
    def events(self) -> Sequence[SpanEvent]: ...

    @property
    def links(self) -> Sequence[SpanLink]: ...


class Span(Protocol):
    def get_context(self) -> SpanContext: ...

    @property
    def is_recording(self) -> bool: ...

    def end(self, end_time: Timestamp | None = None): ...

    def update_name(self, name: str): ...

    def set_status(self, status: SpanStatusValue, description: str | None = None): ...

    def set_attribute(self, key: str, value: AttributeValue): ...

    def set_attributes(self, attributes: Attributes): ...

    def add_event(
        self,
        name: str,
        attributes: Attributes | None = None,
        timestamp: Timestamp | None = None,
    ): ...

    def record_exception(
        self,
        error: BaseException,
        attributes: Attributes | None = None,
        timestamp: Timestamp | None = None,
    ): ...

    def add_link(
        self,
        context: SpanContext,
        attributes: Attributes | None = None,
    ): ...


class SpanProcessor(Protocol):
    def on_start(
        self,
        span: ReadableSpan,
        parent_context: SpanContext | None = None,
    ) -> None: ...
    def on_end(self, span: ReadableSpan) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class SpanExporter(Protocol):
    def export(self, spans: Sequence[ReadableSpan]) -> bool: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class Tracer(Protocol):
    @property
    def is_closed(self) -> bool: ...

    def start_span(
        self,
        name: str,
        parent: SpanContext | None = None,
        attributes: Attributes | None = None,
        links: Sequence[SpanLink] | None = None,
        start_time: Timestamp | None = None,
    ) -> Span: ...

    @contextmanager
    def span(
        self,
        name: str,
        context: SpanContext | None = None,
        attributes: Attributes | None = None,
        links: Sequence[SpanLink] | None = None,
        start_time: Timestamp | None = None,
        record_exception: bool = True,
    ) -> Iterator[Span]: ...

    def add_span_processor(self, processor: SpanProcessor) -> None: ...

    def close(self) -> None: ...


_current_span = ContextVar[Span | None]("current_span", default=None)


def get_current_span() -> Span | None:
    return _current_span.get()


def set_current_span(span: Span | None) -> Token[Span | None]:
    return _current_span.set(span)


def reset_current_span(token: Token[Span | None]) -> None:
    _current_span.reset(token)


_current_trace_id = ContextVar[str | None]("current_trace_id", default=None)


def get_current_trace_id() -> str | None:
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> Token[str | None]:
    return _current_trace_id.set(trace_id)


def reset_current_trace_id(token: Token[str | None]) -> None:
    _current_trace_id.reset(token)


@contextmanager
def use_span(
    span: Span,
    record_exception: bool = True,
) -> Iterator[Span]:
    token = set_current_span(span)
    try:
        yield span
    except Exception as e:
        span.set_status(SpanStatusValue.ERROR)
        if record_exception:
            span.record_exception(e)
    finally:
        span.end()
        reset_current_span(token)


@dataclass
class _Span(Span):
    tracer: _Tracer
    name: str
    context: SpanContext
    parent: SpanContext | None
    status: SpanStatus
    attributes: MutableMapping[str, AttributeValue]
    start_time: Timestamp
    end_time: Timestamp | None
    events: MutableSequence[SpanEvent]
    links: MutableSequence[SpanLink]

    def get_context(self) -> SpanContext:
        return self.context

    @property
    def is_recording(self) -> bool:
        return True

    def end(self, end_time: Timestamp | None = None):
        if self.end_time is not None:
            raise RuntimeError("Span has already ended.")
        self.end_time = end_time or now()
        self.tracer.on_span_end(self)

    def update_name(self, name: str):
        self.name = name

    def set_status(self, status: SpanStatusValue, description: str | None = None):
        self.status = SpanStatus(value=status, description=description)

    def set_attribute(self, key: str, value: AttributeValue):
        self.attributes[key] = value

    def set_attributes(self, attributes: Attributes):
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        attributes: Attributes | None = None,
        timestamp: Timestamp | None = None,
    ):
        event = SpanEvent(
            name=name,
            attributes=attributes or {},
            timestamp=timestamp or now(),
        )
        self.events.append(event)

    def record_exception(
        self,
        error: BaseException,
        attributes: Attributes | None = None,
        timestamp: Timestamp | None = None,
    ):
        message = str(error)
        if hasattr(error, "__notes__") and error.__notes__:
            message += "\n" + "\n".join(
                [f"Note {i}:\n{n}" for i, n in enumerate(error.__notes__)]
            )
        etype, value, tb = sys.exc_info()
        tb_message = traceback.format_exception(etype, value, tb)
        event_attributes: MutableAttributes = {
            "exception.type": type(error).__name__,
            "exception.message": message,
            "exception.stacktrace": "".join(tb_message),
        }
        if attributes:
            event_attributes.update(attributes)
        self.add_event(
            name="exception",
            attributes=event_attributes,
            timestamp=timestamp,
        )

    def add_link(
        self,
        context: SpanContext,
        attributes: Attributes | None = None,
    ):
        link = SpanLink(
            context=context,
            attributes=attributes,
        )
        self.links.append(link)


def new_trace_id() -> str:
    return str(uuid4())


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[None]:
    if trace_id is None:
        trace_id = new_trace_id()
    token = set_current_trace_id(trace_id)
    try:
        yield
    finally:
        reset_current_trace_id(token)


def _new_span_id() -> str:
    return str(uuid4())


class _Tracer:
    _closed: bool
    _processors: list[SpanProcessor]

    def __init__(self) -> None:
        self._closed = False
        self._processors = []

    @property
    def is_closed(self) -> bool:
        return self._closed

    def start_span(
        self,
        name: str,
        parent: SpanContext | None = None,
        attributes: Attributes | None = None,
        links: Sequence[SpanLink] | None = None,
        start_time: Timestamp | None = None,
    ) -> Span:
        if self._closed:
            raise RuntimeError("Tracer is closed.")

        if parent is None:
            parent_span = get_current_span()
            parent = parent_span.get_context() if parent_span is not None else None
        if parent is not None:
            trace_id = parent.trace_id
        else:
            trace_id = get_current_trace_id()
        if trace_id is None:
            raise RuntimeError("No trace ID available for new span.")

        span_id = _new_span_id()

        _attributes = {}
        if attributes is not None:
            _attributes.update(attributes)

        _links = []
        if links is not None:
            _links.extend(links)

        span = _Span(
            tracer=self,
            name=name,
            context=SpanContext(
                trace_id=trace_id,
                span_id=span_id,
            ),
            parent=parent,
            status=SpanStatus(),
            attributes=_attributes,
            start_time=start_time or now(),
            end_time=None,
            events=[],
            links=_links,
        )

        self.on_span_start(span, parent)

        return span

    @contextmanager
    def span(
        self,
        name: str,
        context: SpanContext | None = None,
        attributes: Attributes | None = None,
        links: Sequence[SpanLink] | None = None,
        start_time: Timestamp | None = None,
        record_exception: bool = True,
    ) -> Iterator[Span]:
        span = self.start_span(
            name=name,
            parent=context,
            attributes=attributes,
            links=links,
            start_time=start_time,
        )
        token = set_current_span(span)
        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatusValue.ERROR)
            if record_exception:
                print(f"Recording exception in span: {e}\n{span}")
                span.record_exception(e)
            raise
        finally:
            span.end()
            reset_current_span(token)

    def on_span_start(
        self, span: ReadableSpan, parent_context: SpanContext | None = None
    ) -> None:
        for processor in self._processors:
            processor.on_start(span, parent_context)

    def on_span_end(self, span: ReadableSpan) -> None:
        for processor in self._processors:
            processor.on_end(span)

    def add_span_processor(self, processor: SpanProcessor) -> None:
        if self._closed:
            raise RuntimeError("Tracer is closed.")
        self._processors.append(processor)

    def close(self) -> None:
        self._closed = True


tracer: Tracer = _Tracer()
