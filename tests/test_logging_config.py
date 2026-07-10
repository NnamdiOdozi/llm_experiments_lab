import logging

from backend.logging_config import chatbot_log


def test_chatbot_log_is_a_lab_logger():
    assert chatbot_log.name == "lab.chatbot"
    assert isinstance(chatbot_log, logging.Logger)
