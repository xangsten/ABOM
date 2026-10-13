"""Проверка сборки графа."""
from abom.graph import build_graph


def test_graph_compiles():
    app = build_graph()
    assert hasattr(app, "ainvoke")


def test_graph_has_expected_nodes():
    app = build_graph()
    nodes = set(app.get_graph().nodes)
    assert {"organ", "merge"} <= nodes
