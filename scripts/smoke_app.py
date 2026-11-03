"""Exercise the interactive forecast, outage, time-step and sensor controls."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    test = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not test.exception, test.exception
    assert len(test.metric) >= 5
    test.selectbox(key="model_select").select("graph_robust_ridge").run()
    assert not test.exception, test.exception
    test.selectbox(key="outage_select").select("20%").run()
    assert not test.exception, test.exception
    test.slider(key="test_offset").set_value(17).run()
    assert not test.exception, test.exception
    test.slider(key="sensor_select").set_value(5).run()
    assert not test.exception, test.exception
    assert len(test.get("plotly_chart")) >= 2
    print("Streamlit AppTest passed model, outage, test-offset and sensor interactions.")


if __name__ == "__main__":
    main()
