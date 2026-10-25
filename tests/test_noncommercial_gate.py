from phonon_onnx_bundles.catalog import model_spec

def test_31m_is_commercially_permissive(): assert model_spec("phonon-31m-en")["commercial_use"] is True
def test_40m_is_marked_noncommercial(): assert model_spec("phonon-40m-fr")["commercial_use"] is False
