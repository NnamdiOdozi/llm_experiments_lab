from config.settings import Settings


def test_chatbot_settings_have_expected_defaults():
    s = Settings(nebius_key=None)
    assert s.token_factory_base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert s.token_factory_model == "Qwen/Qwen3-Next-80B-A3B-Thinking"
    assert s.chatbot_log_tail_lines == 50
    assert s.chatbot_history_window_turns == 10
    assert s.nebius_key is None
