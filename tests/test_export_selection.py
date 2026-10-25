from pathlib import Path
from scripts.export_phonon import _select_graph

def test_q4_keeps_text_and_mimi_fp32(tmp_path:Path):
    for n in ("text_conditioner","mimi_decoder"):
        (tmp_path/f"{n}.onnx").write_bytes(b"x")
        assert _select_graph(tmp_path,n,"q4").name==f"{n}.onnx"

def test_q4_uses_quantized_flow(tmp_path:Path):
    (tmp_path/"flow_lm_main_q4.onnx").write_bytes(b"x")
    assert _select_graph(tmp_path,"flow_lm_main","q4").name=="flow_lm_main_q4.onnx"
