# pthread_trace
This tool allows capturing pthread events into a [perfetto](https://perfetto.dev/) trace, which allows visualizing pthread synchronization events on a timeline for debugging and optimization purposes.

To use it, compile the tool to a shared library:
```
c++ -shared -fPIC -ldl -O2 -std=c++14 pthread_trace.cc -o pthread_trace.so
```

Then, run your program with `pthread_trace.so` as an LD_PRELOAD:
```
PTHREAD_TRACE_PATH=trace.proto LD_PRELOAD=$(pwd)/pthread_trace.so <pthread using program> <program arguments...>
```

Now, navigate to [ui.perfetto.dev](https://ui.perfetto.dev), and load the resulting trace.proto file.

## Traced events
This tool captures the following information:
- Time threads are blocked in `pthread_mutex_lock`
- Time threads are blocked in `pthread_cond_wait`/`pthread_cond_timedwait`
- Time threads are blocked in `pthread_join`
- Time threads hold a mutex (i.e. time between `pthread_mutex_lock`/successful `pthread_mutex_trylock` and `pthread_mutex_unlock`)
- `pthread_cond_broadcast` and `pthread_cond_signal` events

## Performance impact
This tool is designed to have as minimal impact on performance as possible.
However, even very lightweight tracing will be significant compared to uncontended thread synchronization primitives.
It is possible (likely even) that enabling this tracing tool will alter the behavior of programs using pthreads.
It might only slow the program down, but it might also dramatically alter the behavior of synchronization primitives.
