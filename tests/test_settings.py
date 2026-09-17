from config.settings import Settings


def test_chatbot_settings_have_expected_defaults():
    s = Settings(nebius_key=None)
    assert s.token_factory_base_url == "https://api.tokenfactory.nebius.com/v1/"
    # Deliberately NOT pinned to a literal model id: Token Factory retires
    # models without notice (two picks 404'd this way), and a test that fails
    # on every catalogue rotation trains you to ignore it. Assert the shape.
    assert s.token_factory_model
    assert "/" in s.token_factory_model  # "provider/model"
    assert s.chatbot_log_tail_lines == 50
    assert s.chatbot_history_window_turns == 10
    assert s.nebius_key is None
