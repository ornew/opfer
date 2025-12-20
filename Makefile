.PHONY: fmt
fmt:
	uvx ruff format

.PHONY: lint
lint:
	uvx ruff check
	uvx pyright

.PHONY: fix
fix:
	uvx ruff check --fix
