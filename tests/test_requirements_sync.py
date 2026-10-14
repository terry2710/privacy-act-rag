from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_requirements(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return set(lines)


def test_gradio_requirements_include_core():
    core = read_requirements(ROOT / "requirements-core.txt")
    gradio = read_requirements(ROOT / "requirements.txt")
    assert core <= gradio, sorted(core - gradio)


def test_streamlit_requirements_include_core():
    core = read_requirements(ROOT / "requirements-core.txt")
    streamlit = read_requirements(ROOT / "demos" / "streamlit" / "requirements.txt")
    assert core <= streamlit, sorted(core - streamlit)
