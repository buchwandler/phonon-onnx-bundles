from pathlib import Path

import pytest

from scripts.export_phonon import _repair_graph_order, _select_graph


def test_q4_keeps_text_and_mimi_fp32(tmp_path:Path):
    for n in ("text_conditioner","mimi_decoder"):
        (tmp_path/f"{n}.onnx").write_bytes(b"x")
        assert _select_graph(tmp_path,n,"q4").name==f"{n}.onnx"

def test_q4_uses_quantized_flow(tmp_path:Path):
    (tmp_path/"flow_lm_main_q4.onnx").write_bytes(b"x")
    assert _select_graph(tmp_path,"flow_lm_main","q4").name=="flow_lm_main_q4.onnx"


def test_repair_graph_order_fixes_float16_style_node_order(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    cast = onnx.helper.make_node("Cast", ["x"], ["cast"], name="cast", to=onnx.TensorProto.FLOAT)
    multiply = onnx.helper.make_node("Mul", ["cast", "x"], ["y"], name="multiply")
    graph = onnx.helper.make_graph(
        [multiply, cast],
        "unsorted",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "unsorted.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)]), path)

    _repair_graph_order(path)

    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert [node.name for node in model.graph.node] == ["cast", "multiply"]
