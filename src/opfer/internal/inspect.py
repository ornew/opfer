from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    TypeGuard,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo


@dataclass
class FuncSchema:
    name: str
    description: str | None
    input_model: type[BaseModel]
    input_type: TypeAdapter[Any]
    input_schema: dict[str, Any]
    output_type: TypeAdapter[Any]
    output_schema: dict[str, Any]
    signature: inspect.Signature
    strict_json_schema: bool = True


def get_func_schema[**I, O](
    func: Callable[I, O],
    name: str | None = None,
    description: str | None = None,
    allow_positional_args: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    func_name = name or func.__name__
    func_description = description or inspect.getdoc(func)

    type_hints_with_extras = get_type_hints(func, include_extras=True)
    type_hints: dict[str, Any] = {}
    param_descs: dict[str, str] = {}

    for name, annotation in type_hints_with_extras.items():
        stripped_ann, metadata = _strip_annotated(annotation)
        type_hints[name] = stripped_ann

        description = _extract_description_from_metadata(metadata)
        if description is not None:
            param_descs.setdefault(name, description)

    sig = inspect.signature(func)

    fields: dict[str, Any] = {}
    params = list(sig.parameters.items())
    takes_context = False

    for i, (name, param) in enumerate(params):
        if not allow_positional_args:
            # キーワード専用 (KEYWORD_ONLY) または **kwargs (VAR_KEYWORD) 以外はエラーにする
            if param.kind not in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"Parameter '{name}' is not a keyword-only argument. "
                    f"Function '{func.__name__}' must strictly use keyword arguments "
                    f"(e.g. def foo(*, x: int)) when allow_positional_args=False."
                )

        ann = type_hints.get(name, param.annotation)
        if ann is inspect.Signature.empty:
            raise ValueError(f"no type annotation found for parameter '{name}'")
        origin = get_origin(ann)
        default = param.default
        field_description = param_descs.get(name)
        match param.kind:
            case param.VAR_POSITIONAL:
                if get_origin(ann) is tuple:
                    # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                    args_of_tuple = get_args(ann)
                    if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                        ann = list[args_of_tuple[0]]
                    else:
                        ann = list[Any]
                else:
                    # If user wrote *args: int, treat as List[int]
                    ann = list[ann]
                fields[name] = (
                    ann,
                    Field(default_factory=list, description=field_description),
                )
            case param.VAR_KEYWORD:
                # **kwargs handling
                if get_origin(ann) is dict:
                    # e.g. def foo(**kwargs: dict[str, int])
                    dict_args = get_args(ann)
                    if len(dict_args) == 2:
                        ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                    else:
                        ann = dict[str, Any]
                else:
                    # e.g. def foo(**kwargs: int) -> Dict[str, int]
                    ann = dict[str, ann]  # type: ignore

                fields[name] = (
                    ann,
                    Field(default_factory=dict, description=field_description),
                )
            case _:
                if default is inspect.Parameter.empty:
                    fields[name] = (ann, Field(..., description=field_description))
                elif isinstance(default, FieldInfo):
                    fields[name] = (
                        ann,
                        FieldInfo.merge_field_infos(
                            default,
                            description=field_description or default.description,
                        ),
                    )
                else:
                    fields[name] = (
                        ann,
                        Field(default=default, description=field_description),
                    )

    input_model = create_model(
        f"{func_name}_input",
        __base__=BaseModel,
        **fields,
    )

    return_annotation = type_hints.get("return", sig.return_annotation)
    if return_annotation is inspect.Signature.empty:
        raise ValueError("no type annotation found for return value")

    input_type = TypeAdapter(input_model)
    output_type = TypeAdapter(return_annotation)

    input_schema = input_model.model_json_schema()
    output_schema = output_type.json_schema()

    if strict_json_schema:
        input_schema = ensure_strict_json_schema(input_schema)
        output_schema = ensure_strict_json_schema(output_schema)

    return FuncSchema(
        name=func_name,
        description=func_description,
        input_model=input_model,
        input_type=input_type,
        input_schema=input_schema,
        output_type=output_type,
        output_schema=output_schema,
        signature=sig,
        strict_json_schema=strict_json_schema,
    )


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns the underlying annotation and any metadata from typing.Annotated."""

    metadata: tuple[Any, ...] = ()
    ann = annotation

    while get_origin(ann) is Annotated:
        args = get_args(ann)
        if not args:
            break
        ann = args[0]
        metadata = (*metadata, *args[1:])

    return ann, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extracts a human readable description from Annotated metadata if present."""

    for item in metadata:
        if isinstance(item, str):
            return item
        elif isinstance(item, FieldInfo):
            if item.description is not None:
                return item.description
    return None


_EMPTY_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}


def ensure_strict_json_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the OpenAI API expects.
    """
    if schema == {}:
        return _EMPTY_SCHEMA
    return _ensure_strict_json_schema(schema, path=(), root=schema)


# Adapted from https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not _is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if _is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    definitions = json_schema.get("definitions")
    if _is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif (
        typ == "object"
        and "additionalProperties" in json_schema
        and json_schema["additionalProperties"]
    ):
        raise ValueError(
            "additionalProperties should not be set for object types. This could be because "
            "you're using an older version of Pydantic, or because you configured additional "
            "properties to be allowed. If you really need this, update the function or output tool "
            "to not use a strict schema." + f" {json_schema}"
        )

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if _is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if _is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # unions
    any_of = json_schema.get("anyOf")
    if _is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # oneOf is not supported by OpenAI's structured outputs in nested contexts,
    # so we convert it to anyOf which provides equivalent functionality for
    # discriminated unions
    one_of = json_schema.get("oneOf")
    if _is_list(one_of):
        existing_any_of = json_schema.get("anyOf", [])
        if not _is_list(existing_any_of):
            existing_any_of = []
        json_schema["anyOf"] = existing_any_of + [
            _ensure_strict_json_schema(
                variant, path=(*path, "oneOf", str(i)), root=root
            )
            for i, variant in enumerate(one_of)
        ]
        json_schema.pop("oneOf")

    # intersections
    all_of = json_schema.get("allOf")
    if _is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    # if json_schema.get("default") is None:
    #     json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and _has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = _resolve_ref(root=root, ref=ref)
        if not _is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert _is_dict(
            value
        ), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved


def _is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


def _is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False
