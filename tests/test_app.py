from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_signals_forecast_month_and_empty_state():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=60)
    assert not app.exception
    assert len(app.metric) == 4
    assert [m.label for m in app.metric[1:]] == ['부족 가능성 프로젝트', '여유 가능성 프로젝트', '판단 유보 프로젝트']
    assert app.selectbox[0].options == ['2026-07', '2026-08', '2026-09']
    app.selectbox[0].select('2026-09').run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '500.0'
    assert {'판단', '근거', '전달 평균 OT(시간/FTE)'}.issubset(app.dataframe[0].value.columns)
    assert {'이전월 유지', 'Random Forest', 'XGBoost', '관측 OT(시간/FTE)'}.issubset(app.dataframe[1].value.columns)
    assert {'모델 결정', '누적 후보 학습 건수', '후보 MAE(시간/FTE)'}.issubset(app.dataframe[3].value.columns)
    for item in app.dataframe:
        assert '부족 FTE' not in item.value.columns
        assert '평균 추정 FTE' not in item.value.columns
    app.number_input[0].set_value(25.0).run(timeout=60)
    assert not app.exception
    app.checkbox[0].uncheck().run()
    assert not app.exception
    assert not app.metric
