import copy


def _add_adjusted_col_offsets(table):
    """Add adjusted column offsets to take into account multi-column cells."""
    adjusted_table = []
    for row in table:
        real_col_index = 0
        adjusted_row = []
        for cell in row:
            adjusted_cell = copy.deepcopy(cell)
            adjusted_cell["adjusted_col_start"] = real_col_index
            adjusted_cell["adjusted_col_end"] = (
                adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"]
            )
            real_col_index += adjusted_cell["column_span"]
            adjusted_row.append(adjusted_cell)
        adjusted_table.append(adjusted_row)
    return adjusted_table


def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
    """Heuristic to find row headers."""
    row_headers = []
    row = adjusted_table[row_index]
    for i in range(0, col_index):
        if row[i]["is_header"]:
            row_headers.append(row[i])
    return row_headers


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
    """Heuristic to find column headers."""
    adjusted_cell = adjusted_table[row_index][col_index]
    adjusted_col_start = adjusted_cell["adjusted_col_start"]
    adjusted_col_end = adjusted_cell["adjusted_col_end"]
    col_headers = []
    for r in range(0, row_index):
        row = adjusted_table[r]
        for cell in row:
            if (
                cell["adjusted_col_start"] < adjusted_col_end
                and cell["adjusted_col_end"] > adjusted_col_start
            ):
                if cell["is_header"]:
                    col_headers.append(cell)

    return col_headers


def get_highlighted_subtable(table, cell_indices, with_heuristic_headers=False):
    """Extract out the highlighted part of a table."""
    highlighted_table = []

    adjusted_table = _add_adjusted_col_offsets(table)

    for row_index, col_index in cell_indices:
        cell = table[row_index][col_index]

        if with_heuristic_headers:
            row_headers = _get_heuristic_row_headers(
                adjusted_table, row_index, col_index
            )
            col_headers = _get_heuristic_col_headers(
                adjusted_table, row_index, col_index
            )
        else:
            row_headers = []
            col_headers = []

        highlighted_cell = {
            "cell": cell,
            "row_headers": row_headers,
            "col_headers": col_headers,
        }
        highlighted_table.append(highlighted_cell)

    return highlighted_table


def process_subtable(subtable, table_page_title, table_section_title):
    """Process the highlighted subtable and return a list of features."""
    parent_table = []
    source = []

    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]

        if cell["is_header"]:
            parent_table.append(("header", cell["value"]))
            source.append(f"{table_page_title} | header | {cell['value']}")
        else:
            if col_headers == [] and row_headers == []:
                # cell
                parent_table.append((table_section_title, cell["value"]))
                source.append(
                    f'{table_page_title} | {table_section_title} | {cell["value"]}'
                )
            else:
                for col_header in col_headers:
                    # col | cell
                    parent_table.append((col_header["value"], cell["value"]))
                    source.append(
                        f'{table_page_title} | {col_header["value"]} | {cell["value"]}'
                    )

    return parent_table, source


def camel_case_to_natural_text(s: str):
    # https://stackoverflow.com/a/44969381
    # Example:
    # [ "Aarhus_Airport | elevationAboveTheSeaLevel | 25.0" ]
    # -> [ "Aarhus Airport | elevation above the sea level | 25.0" ]
    return "".join([" " + c.lower() if c.isupper() else c for c in s]).lstrip(" ")