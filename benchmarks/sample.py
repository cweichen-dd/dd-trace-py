import ddtrace.auto

import os

from ddtrace import tracer

LOOPS = 5 if os.getenv("WARMUP", "0") == "1" else 100_000
LOOPS = 5

for i in range(LOOPS):
    with tracer.trace("root"):
        for _ in range(20):
            with tracer.trace("child") as span:
                span.set_tag("foo", "bar")
                span.set_tag("baz", "qux")
                span.set_tag("quux", "corge")
                span.set_tag("grault", "garply")
    if i % 1000 == 0:
        tracer.flush()
