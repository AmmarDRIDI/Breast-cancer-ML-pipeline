from src.train import load_data


def test_split_sizes_and_feature_count():
    x_train, x_val, x_test, y_train, y_val, y_test, feature_names = load_data(
        seed=42)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_test) == len(y_test)

    total = len(x_train) + len(x_val) + len(x_test)
    assert total == 569
    assert len(feature_names) == 30
