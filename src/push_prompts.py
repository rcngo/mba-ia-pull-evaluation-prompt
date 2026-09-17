import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"
EXPECTED_VARIABLES = {"bug_report"}


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    errors = []

    for field in ("description", "system_prompt", "version"):
        if field not in prompt_data:
            errors.append(f"Campo obrigatório faltando: {field}")

    system_prompt = (prompt_data.get("system_prompt") or "").strip()

    if not system_prompt:
        errors.append("system_prompt está vazio")

    if "TODO" in system_prompt:
        errors.append("system_prompt ainda contém TODOs")

    user_prompt = (prompt_data.get("user_prompt") or "").strip()

    if not user_prompt:
        errors.append("user_prompt está vazio")

    techniques = prompt_data.get("techniques_applied", [])

    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("human", prompt_data["user_prompt"]),
    ])


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    is_valid, errors = validate_prompt(prompt_data)

    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("   ✓ Estrutura do prompt validada")

    try:
        chat_prompt = build_chat_prompt(prompt_data)
    except Exception as e:
        print(f"❌ Erro ao montar o ChatPromptTemplate: {e}")
        print("   Verifique se todas as chaves literais no texto estão escapadas como chave dupla.")
        return False

    variables = set(chat_prompt.input_variables)

    if variables != EXPECTED_VARIABLES:
        print(f"❌ Variáveis inesperadas no template: {sorted(variables)}")
        print(f"   O avaliador injeta apenas: {sorted(EXPECTED_VARIABLES)}")
        return False

    print(f"   ✓ Variáveis do template: {sorted(variables)}")

    techniques = prompt_data.get("techniques_applied", [])
    description = f"{prompt_data['description']} | Técnicas: {', '.join(techniques)}"

    try:
        client = Client()
        url = client.push_prompt(
            prompt_name,
            object=chat_prompt,
            is_public=True,
            description=description,
            tags=list(prompt_data.get("tags", [])),
        )
    except Exception as e:
        if "nothing to commit" in str(e).lower():
            print("   ✓ Conteúdo idêntico ao último commit, nada a publicar")
            print(f"   ✓ Metadados e visibilidade pública atualizados em {prompt_name}")
            return True

        print(f"❌ Erro ao publicar prompt no LangSmith: {e}")
        return False

    print(f"   ✓ Publicado: {url}")

    return True


def main():
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    prompts = load_yaml(PROMPTS_FILE)

    if not prompts:
        return 1

    prompt_data = prompts.get(PROMPT_KEY)

    if not prompt_data:
        print(f"❌ Chave '{PROMPT_KEY}' não encontrada em {PROMPTS_FILE}")
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    prompt_name = f"{username}/{PROMPT_KEY}"

    print(f"Publicando: {prompt_name}")
    print(f"Técnicas aplicadas: {', '.join(prompt_data.get('techniques_applied', []))}\n")

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n✅ Push concluído com sucesso")
    print("\nPróximo passo: python src/evaluate.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
