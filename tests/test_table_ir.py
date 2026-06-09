from __future__ import annotations

from src.pipeline.table_ir import parse_html_table


def test_parse_html_table_selects_largest_table_and_expands_spans() -> None:
    html = (
        "<table><tr><td>x</td></tr></table>"
        "<table>"
        "<caption>主表</caption>"
        "<tr><th rowspan='2'>地区</th><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>10</td><td>20</td></tr>"
        "</table>"
    )

    table_ir = parse_html_table(html)

    assert table_ir.parse_valid is True
    assert table_ir.has_table is True
    assert table_ir.caption == "主表"
    assert table_ir.row_count == 2
    assert table_ir.col_count == 3
    assert len(table_ir.cells) == 5
    assert table_ir.grid[0][0] == table_ir.grid[1][0]


def test_parse_html_table_corrects_invalid_spans_to_one() -> None:
    html = "<table><tr><td rowspan='0' colspan='-3'>A</td><td>B</td></tr></table>"

    table_ir = parse_html_table(html)

    assert table_ir.parse_valid is True
    assert table_ir.has_table is True
    assert table_ir.cells[0].rowspan == 1
    assert table_ir.cells[0].colspan == 1
    assert table_ir.diagnostics["invalid_spans"]
