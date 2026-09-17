import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPTS_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt():
    prompts = load_prompts(PROMPTS_FILE)
    assert prompts is not None, f"Não foi possível carregar {PROMPTS_FILE}"
    assert PROMPT_KEY in prompts, f"Chave '{PROMPT_KEY}' ausente em {PROMPTS_FILE}"
    return prompts[PROMPT_KEY]


@pytest.fixture(scope="module")
def full_text(prompt):
    return f"{prompt.get('system_prompt', '')}\n{prompt.get('user_prompt', '')}"


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        assert "system_prompt" in prompt, "Campo 'system_prompt' não existe"

        system_prompt = prompt["system_prompt"]

        assert isinstance(system_prompt, str), "'system_prompt' deve ser uma string"
        assert system_prompt.strip(), "'system_prompt' está vazio"
        assert len(system_prompt.strip()) > 200, "'system_prompt' é curto demais para um prompt otimizado"

    def test_prompt_has_role_definition(self, prompt):
        system_prompt = prompt.get("system_prompt", "")

        assert re.search(
            r"você é (um|uma)\s+\w+", system_prompt, re.IGNORECASE
        ), "O prompt não define uma persona no formato 'Você é um/uma ...'"

        personas = ["product owner", "product manager", "engenheiro", "analista", "especialista"]

        assert any(p in system_prompt.lower() for p in personas), (
            f"A persona definida não é uma das esperadas: {personas}"
        )

    def test_prompt_mentions_format(self, full_text):
        text = full_text.lower()

        assert "como um" in text and "eu quero" in text and "para que" in text, (
            "O prompt não exige o template de User Story padrão (Como um... eu quero... para que...)"
        )

        assert "critérios de aceitação" in text, (
            "O prompt não exige uma seção de Critérios de Aceitação"
        )

        assert all(c in text for c in ("dado que", "quando", "então")), (
            "O prompt não exige critérios no formato Dado/Quando/Então"
        )

    def test_prompt_has_few_shot_examples(self, prompt):
        system_prompt = prompt.get("system_prompt", "")
        lower = system_prompt.lower()

        assert "exemplo" in lower, "O prompt não contém exemplos (técnica Few-shot)"

        assert lower.count("entrada:") >= 2, (
            "São esperados ao menos 2 exemplos com 'Entrada:' demonstrando o input"
        )

        assert lower.count("saída:") >= 2, (
            "São esperados ao menos 2 exemplos com 'Saída:' demonstrando o output"
        )

        assert system_prompt.count("Como um") >= 2, (
            "Os exemplos few-shot devem demonstrar o formato de User Story na saída"
        )

    def test_prompt_no_todos(self, prompt):
        for field, value in prompt.items():
            if not isinstance(value, str):
                continue

            assert "[TODO]" not in value, f"Campo '{field}' ainda contém [TODO]"
            assert "TODO" not in value, f"Campo '{field}' ainda contém TODO"
            assert "FIXME" not in value, f"Campo '{field}' ainda contém FIXME"

    def test_minimum_techniques(self, prompt):
        techniques = prompt.get("techniques_applied")

        assert techniques is not None, "Metadado 'techniques_applied' ausente no YAML"
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

        assert any("few-shot" in t.lower() for t in techniques), (
            "Few-shot Learning é obrigatório e deve estar listado em 'techniques_applied'"
        )

        is_valid, errors = validate_prompt_structure(prompt)

        assert is_valid, f"Estrutura do prompt inválida: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
