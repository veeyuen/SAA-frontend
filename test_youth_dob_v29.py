import pandas as pd
from pathlib import Path
import ast
import json

SOURCE = Path(__file__).resolve().parent / 'streamlit_BQ_29_DOB_BOUNDS_FIX.py'
NOTEBOOK = Path(__file__).resolve().parent / 'SEA_YOUTH_PRODUCTION_v14_DOB_BOUNDS_FIX.ipynb'

def isolated_function(text, name):
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'pd': pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<isolated>', 'exec'), ns)
    return ns[name]

def test_parsers_agree():
    script = isolated_function(SOURCE.read_text(), '_youth_2026_parse_dob_series')
    nb = json.loads(NOTEBOOK.read_text())
    cell = ''.join(nb['cells'][5]['source'])
    book = isolated_function(cell, 'parse_dob_series')
    inputs = pd.Series([
        '2009-03-06', '2010-12-05', '2010-04-05 00:00:00+00:00',
        '2009-05-11', '2009-11-05', '05/04/2010', '04-12-2008',
        '4/12/08', '', '-', '<NA>', None, '2010', '2008-12-05',
    ], dtype='object')
    expected = [
        '2009-03-06','2010-12-05','2010-04-05','2009-05-11',
        '2009-11-05','2010-04-05','2008-12-04','2008-12-04',
        None,None,None,None,'2010-01-01','2008-12-05',
    ]
    for parser in (script,book):
        got = parser(inputs).dt.strftime('%Y-%m-%d').tolist()
        for actual, intended in zip(got,expected):
            assert (actual if not pd.isna(actual) else None) == intended, (actual,intended)
    pd.testing.assert_series_equal(script(inputs),book(inputs))

def test_index_preservation():
    fn = isolated_function(SOURCE.read_text(), '_youth_2026_parse_dob_series')
    s = pd.Series(['2010-04-05','05/04/2010'], index=[42,99])
    r = fn(s)
    assert r.index.tolist() == [42,99]
    assert r.iloc[0] == r.iloc[1]


def test_out_of_bounds_dob_and_iso_timestamp():
    fn = isolated_function(SOURCE.read_text(), '_youth_2026_parse_dob_series')
    nb = json.loads(NOTEBOOK.read_text())
    book = isolated_function(''.join(nb['cells'][5]['source']), 'parse_dob_series')
    s = pd.Series(['7545-08-04 00:00:00', '04/08/7545',
                   '9999-01-01', '1800-01-01', '2030-02-03',
                   '2009-03-06 00:00:00+00:00', '05/04/2010',
                   '2010-12-05', 'not a date'])
    for parser in [fn,book]:
        got = parser(s).dt.strftime('%Y-%m-%d').tolist()
        normalized = [None if pd.isna(x) else x for x in got]
        assert normalized == [None,None,None,None,None,
                              '2009-03-06','2010-04-05','2010-12-05',None]
