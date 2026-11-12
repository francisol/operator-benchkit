from unittest.mock import patch

from benchkit.core import Case, measure, run_cases


def test_case_record():
    case = Case(fn=lambda: None, check=lambda: None, logical_bytes=64, info={'k': 8})
    assert case.logical_bytes == 64
    assert case.info['k'] == 8


def test_measure_graph_eager_with_fake_cuda():
    # Stub PyTorch API so unit tests remain runnable on GitHub-hosted CPU runners.
    class Event:
        def __init__(self, enable_timing):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 2.0

    class Graph:
        def replay(self):
            pass

    class GraphContext:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    class Cuda:
        @staticmethod
        def synchronize():
            pass
        @staticmethod
        def graph(graph):
            return GraphContext()

    Cuda.Event = Event
    Cuda.CUDAGraph = Graph

    import sys
    from types import SimpleNamespace
    fake_torch = SimpleNamespace(cuda=Cuda())
    with patch.dict(sys.modules, {'torch': fake_torch}):
        for mode in ('eager', 'graph'):
            calls = []
            median, samples = measure(lambda: calls.append(1), mode=mode, warmup=2,
                                      graph_iters=5, repeat=4, rounds=3)
            assert len(samples) == 3
            assert median == (500.0 if mode == 'eager' else 100.0)
