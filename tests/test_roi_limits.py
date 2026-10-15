from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest
from roi_limits import assess_limits, read_input, LABELS


def rows():
    return pd.DataFrame([
        ['2026-10', '001', 'Cell', 12.5, np.nan],
        ['2026-10', '001', 'Pack', 14., 15.],
        ['2026-11', '001', 'Cell', np.nan, 0.],
        ['2026-11', '002', 'Cell', 0., 0.],
    ], columns=LABELS)


def test_limits_overrides_missing_and_zero():
    out = assess_limits(rows(), 11., 14.)
    assert out.excess_fte.iloc[0] == 1.5
    assert out.remaining_fte.iloc[1] == 1.
    assert out.limit_source.iloc[1] == '프로젝트별 입력'
    assert out.status.iloc[2] == '투입 FTE 미입력'
    assert pd.isna(out.excess_fte.iloc[2])
    assert out.effective_limit_fte.iloc[2] == 0.
    assert out.status.iloc[3] == '한도 이내'
    assert out.scope.eq('개발조직만').all()
    assert assess_limits(rows(), 13., 14.).excess_fte.iloc[0] == 0.


@pytest.mark.parametrize('column,value', [('development_fte', -1), ('development_fte', 'bad'),
    ('development_fte', float('inf')), ('limit_fte', -1), ('family', 'Other'),
    ('month', '2026-13'), ('month', '2026-1'), ('project_id', ' ')])
def test_invalid_inputs(column, value):
    data = rows().astype(object)
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        assess_limits(data, 11., 14.)


def test_duplicate_rows_and_missing_columns():
    data = rows()
    with pytest.raises(ValueError, match='한 행'):
        assess_limits(pd.concat([data, data.iloc[:1]]), 11., 14.)
    with pytest.raises(ValueError, match='필수 열'):
        assess_limits(data.drop(columns='development_fte'), 11., 14.)
    assert len(assess_limits(data.drop(columns='limit_fte'), 11., 14.)) == 4


def test_excel_and_csv_read_identical_and_preserve_ids():
    # A tiny in-memory fixture exercises the actual XLSX reader, not a mock.
    wb = Workbook()
    ws = wb.active
    ws.append(list(LABELS.values()))
    ws.append(['2026-10', '001', 'Cell', 12.5, 11.])
    file = BytesIO(); wb.save(file)
    excel = read_input(file.getvalue(), 'input.xlsx')
    csv = read_input(excel.rename(columns=LABELS).to_csv(index=False).encode('utf-8-sig'), 'input.csv')
    pd.testing.assert_frame_equal(assess_limits(excel, 11., 14.), assess_limits(csv, 11., 14.))
    assert excel.project_id.iloc[0] == '001'
    assert assess_limits(excel, 11., 14.).excess_fte.iloc[0] == 1.5


def test_standalone_page_needs_no_hr_data():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'pages/3_roi_limits.py').run()
    assert not app.exception
    assert app.title[0].value == '개발조직 FTE 한도'
    assert len(app.number_input) == 2
    app.checkbox[0].check().run()
    assert not app.exception
    assert '직접 행을 추가' in app.info[0].value
    app.session_state['roi_manual'] = {'edited_rows': {}, 'added_rows': [
        {'month': '2026-10', 'project_id': '001', 'family': 'Cell', 'development_fte': 12.5}], 'deleted_rows': []}
    app.run()
    assert not app.exception
    assert app.dataframe[-1].value['초과 FTE'].iloc[0] == 1.5
