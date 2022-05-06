def test_base():
    import housing_price
    import housing_price.ingest_data as data
    import housing_price.score as score
    import housing_price.train as train
    from housing_price.logger import configure_logger
