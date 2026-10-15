from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_team_page_models_and_forecast():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'pages/2_team_models.py')).run(timeout=120)
    assert not app.exception
    assert len(app.tabs)==4
    assert app.selectbox[0].value=='2026-06'
    assert any('p50' in item.value.columns for item in app.dataframe)
    assert any('selected_algorithm' in item.value.columns for item in app.dataframe)
    app.checkbox[0].uncheck().run(timeout=30)
    assert not app.exception
    assert not app.dataframe
