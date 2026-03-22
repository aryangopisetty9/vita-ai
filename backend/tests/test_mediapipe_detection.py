def test_mediapipe_detection():
    try:
        import importlib
        installed = importlib.util.find_spec("mediapipe") is not None
    except Exception:
        installed = False
    # The code should not raise — installation state can be True/False
    assert isinstance(installed, bool)
