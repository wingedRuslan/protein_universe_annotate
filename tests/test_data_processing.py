import pandas as pd
from protein_universe_annotate.data_processing import select_long_tail


def test_select_long_tail():
    # Create a test DataFrame
    data = {
        'col1': ['A'] * 10 + ['B'] * 30 + ['C'] * 60,
        'col2': list(range(100))
    }
    df = pd.DataFrame(data)

    # Test with col1
    selected_df = select_long_tail(df, 'col1')
    assert len(selected_df) == 90
    assert len(selected_df[selected_df['col1'] == 'A']) == 10
    assert len(selected_df[selected_df['col1'] == 'B']) == 30
    assert len(selected_df[selected_df['col1'] == 'C']) == 50

    # Test with col2
    selected_df = select_long_tail(df, 'col2')
    assert len(selected_df) == 100
