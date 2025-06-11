from bm.utils import drop_traces
from ddtrace import tracer

import pyperf

drop_traces(tracer)


def nested_spans_with_set_tag(loops: int) -> float:
    t0 = pyperf.perf_counter()
    for _ in range(loops):
        with tracer.trace("root"):
            with tracer.trace("child1") as span:
                span.set_tag("key", "value")
            with tracer.trace("child2") as span:
                span.set_tag("key", "value")
    t1 = pyperf.perf_counter()
    return t1 - t0


def nested_spans(loops: int) -> float:
    t0 = pyperf.perf_counter()
    for _ in range(loops):
        with tracer.trace("root"):
            with tracer.trace("child1"):
                pass
            with tracer.trace("child2"):
                pass
    t1 = pyperf.perf_counter()
    return t1 - t0


def root_span_only(loops: int) -> float:
    t0 = pyperf.perf_counter()
    for _ in range(loops):
        with tracer.trace("root"):
            pass
    t1 = pyperf.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.metadata["description"] = "Benchmark for creating simple traces."
    runner.bench_time_func("bench_nested_spans_with_set_tag", nested_spans_with_set_tag)
    runner.bench_time_func("bench_nested_spans", nested_spans)
    runner.bench_time_func("bench_root_span_only", root_span_only)
