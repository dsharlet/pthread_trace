# pthread_trace
This tool allows capturing pthread events into a [perfetto](https://perfetto.dev/) trace, which allows visualizing pthread synchronization events on a timeline for debugging and optimization purposes.

To use it, compile the tool to a shared library:
```
g++ -shared -fPIC -ldl -O3 pthread_trace.cc -o pthread_trace.so
```

Then, run your program with `pthread_trace.so` as an LD_PRELOAD:
```
PTHREAD_TRACE_PATH=trace.proto LD_PRELOAD=$(pwd)/pthread_trace.so <pthread using program> <program arguments...>
```

Now, navigate to [ui.perfetto.dev](https://ui.perfetto.dev), and load the resulting trace.proto file.

## Performance debugging
This tool is designed to have as minimal impact on performance as possible.
However, even very lightweight tracing is still going to be significant compared to uncontended thread synchronization primitives.
It is possible (likely even) that enabling this tracing tool will alter the behavior of pthread performance.