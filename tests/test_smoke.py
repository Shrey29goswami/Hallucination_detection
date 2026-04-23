from app.services.guardrails import GuardrailService


def test_guardrails_blocks_toxic_text():
    result = GuardrailService().assess("say hateful thing", "I hate all people and want to kill")
    assert result["overall_risk"] > 0
