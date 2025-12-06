from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Coroutine, Generator, Self, Sequence


class Task[O](Awaitable[O]):
    def __init__(
        self,
        workflow: Workflow,
        task: asyncio.Task,
    ) -> None:
        self._workflow = workflow
        self._task = task

    def __await__(self) -> Generator[Any, None, O]:
        yield from self._task.__await__()
        return self._task.result()

    def then[P](self, fn: Callable[[O], Awaitable[P]]) -> Task[P]:
        async def wrapper() -> P:
            result = await self
            return await fn(result)

        return self._workflow.run(wrapper())

    def then_sync[P](self, fn: Callable[[O], P]) -> Task[P]:
        async def wrapper() -> P:
            result = await self
            return fn(result)

        return self._workflow.run(wrapper())

    def __mul__[P](self, other: Awaitable[P]) -> TaskTuple[O, P]:
        async def wrapper() -> tuple[O, P]:
            result1 = await self
            result2 = await other
            return (result1, result2)

        return self._workflow.run_tuple(wrapper())


class TaskTuple[*O](Awaitable[tuple[*O]]):
    def __init__(
        self,
        workflow: Workflow,
        task: asyncio.Task[tuple[*O]],
    ) -> None:
        self._workflow = workflow
        self._task = task

    def __await__(self) -> Generator[Any, None, tuple[*O]]:
        yield from self._task.__await__()
        return self._task.result()

    def then[P](self, fn: Callable[[*O], Awaitable[P]]) -> Task[P]:
        async def wrapper() -> P:
            results = await self
            return await fn(*results)

        return self._workflow.run(wrapper())

    def then_sync[P](self, fn: Callable[[*O], P]) -> Task[P]:
        async def wrapper() -> P:
            results = await self
            return fn(*results)

        return self._workflow.run(wrapper())

    def __mul__[P](self, other: Awaitable[P]) -> TaskTuple[*O, P]:
        async def wrapper() -> tuple[*O, P]:
            results1 = await self
            result2 = await other
            return (*results1, result2)

        return self._workflow.run_tuple(wrapper())


class Workflow:
    _reminds: list[Task | TaskTuple]

    def __init__(self) -> None:
        self._task_group = asyncio.TaskGroup()
        self._reminds = []

    async def __aenter__(self) -> Self:
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._task_group.__aexit__(exc_type, exc_value, traceback)

    def gather[O](self, a1: Sequence[Awaitable[O]]) -> Task[list[O]]:
        async def wrapper() -> list[O]:
            return await asyncio.gather(*a1)

        task = Task(self, self._task_group.create_task(wrapper()))
        self._reminds.append(task)
        return task

    def run[T1](
        self,
        a1: Coroutine[Any, Any, T1],
        *,
        name: str | None = None,
    ) -> Task[T1]:
        task = Task(self, self._task_group.create_task(a1, name=name))
        self._reminds.append(task)
        return task

    def run_tuple[*T](
        self,
        a1: Coroutine[Any, Any, tuple[*T]],
        *,
        name: str | None = None,
    ) -> TaskTuple[*T]:
        task = TaskTuple(self, self._task_group.create_task(a1, name=name))
        self._reminds.append(task)
        return task


async def example():
    async with Workflow() as wf:
        print("Running tasks...")
        task1 = wf.run(asyncio.sleep(0.5, result=10))
        task1.then_sync(lambda x: print(f"Task1 completed with result: {x}"))
        task2 = wf.run(asyncio.sleep(1.0, result=20))
        task2.then_sync(lambda x: print(f"Task2 completed with result: {x}"))
        task3 = (task1 * task2).then_sync(lambda x, y: x + y)
        task3.then_sync(lambda result: print(f"Task3 completed with result: {result}"))
        print("Waiting for results...")
    print(f"Result: {await task3}")

    async def increment(x: int) -> int:
        return x + 1

    async with Workflow() as wf:
        print("Running tasks...")
        task1 = [wf.run(increment(i)) for i in range(5)]
        task1_all = wf.gather(task1)
        task1_all.then_sync(lambda results: print(f"All increments: {results}"))

        async def delay(x: int) -> int:
            await asyncio.sleep(0.5)
            return x

        task2 = task1_all.then_sync(sum).then(delay)

        task2.then_sync(lambda total: print(f"Sum of increments: {total}"))
        task3 = [i.then(increment) for i in task1]
        task3_all = wf.gather(task3)
        task3_all.then_sync(lambda results: print(f"All double increments: {results}"))

        task4 = task3_all.then_sync(sum)

        (task2 * task3_all * task4).then_sync(
            lambda total, doubles, dtotal: print(
                f"Total: {total}, Double Incs: {doubles}, Double Incs Total: {dtotal}"
            )
        )
        print("Waiting for results...")
        a = await task3_all
        print(f"Double increments: {a}")
        a = await (task2 * task3_all)
        print(f"Sum and double increments: {a}")
    a = await task3_all
    print(f"Double increments: {a}")
    # a = await (task2 * task3_all) # error: already finished TaskGroup
    # print(f"Sum and double increments: {a}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example())
