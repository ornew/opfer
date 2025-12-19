import random
import time
from functools import wraps

from opfer.internal import attributes, events, operations
from opfer.internal.logging import logger
from opfer.internal.tracing import tracer


def retry(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    max_delay: float = 60,
    errors: tuple = (),
):
    def deco(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            with tracer.span(
                operations.OPFER_RETRYABLE_CALL,
                attributes={
                    attributes.OPERATION_NAME: operations.OPFER_RETRYABLE_CALL,
                    attributes.DESCRIPTION: f"retryable call for function: {func.__name__} (max_retries={max_retries})",
                    attributes.FUNCTION_NAME: func.__name__,
                    attributes.RETRY_MAX_COUNT: max_retries,
                },
            ) as s:
                # Initialize variables
                num_retries = 0
                delay = initial_delay

                # Loop until a successful response or max_retries is hit or an exception is raised
                while True:
                    try:
                        if num_retries > 0:
                            s.add_event(
                                events.OPFER_RETRY,
                                attributes={
                                    attributes.EVENT_NAME: events.OPFER_RETRY,
                                    attributes.DESCRIPTION: f"retrying {func.__name__} (attempt {num_retries} of {max_retries}) after {delay:.2f} seconds",
                                    attributes.RETRY_COUNT: num_retries,
                                    attributes.RETRY_DELAY_SECONDS: delay,
                                },
                            )
                        with tracer.span(
                            operations.OPFER_RETRYABLE_CALL_ATTEMPT,
                            attributes={
                                attributes.OPERATION_NAME: operations.OPFER_RETRYABLE_CALL_ATTEMPT,
                                attributes.DESCRIPTION: f"attempting call to function: {func.__name__} (attempt {num_retries} of {max_retries})",
                                attributes.FUNCTION_NAME: func.__name__,
                                attributes.RETRY_COUNT: num_retries,
                                attributes.RETRY_DELAY_SECONDS: delay
                                if num_retries > 0
                                else None,
                            },
                        ):
                            return await func(*args, **kwargs)

                    # Retry on specific errors
                    except errors as e:
                        s.record_exception(
                            e,
                            attributes={
                                attributes.EVENT_NAME: events.OPFER_RETRYABLE_CALL_ERROR,
                                attributes.DESCRIPTION: f"error encountered during call to function: {func.__name__}",
                            },
                        )
                        logger.warning(f"retrying {func.__name__} due to error: {e}")

                        # Increment retries
                        num_retries += 1

                        # Check if max retries has been reached
                        if num_retries > max_retries:
                            raise Exception(
                                f"maximum number of retries ({max_retries}) exceeded."
                            ) from e

                        # Increment the delay
                        if delay < max_delay:
                            delay *= exponential_base
                        delay = min(delay, max_delay) + jitter * random.random()

                        # Sleep for the delay
                        time.sleep(delay)

                    # Raise exceptions for any errors not specified
                    except Exception:
                        raise

        return wrapped

    return deco
