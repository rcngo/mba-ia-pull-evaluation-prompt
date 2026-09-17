import sys
from datetime import date
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

HUB_PROMPT = "leonanluppi/bug_to_user_story_v1"
OUTPUT_FILE = "prompts/bug_to_user_story_v1.yml"
PROMPT_KEY = "bug_to_user_story_v1"


def extract_messages(prompt_template) -> dict:
    messages = {}

    for message in prompt_template.messages:
        template = getattr(getattr(message, "prompt", None), "template", None)

        if template is None:
            continue

        role = type(message).__name__

        if role.startswith("System"):
            messages["system_prompt"] = template
        elif role.startswith("Human"):
            messages["user_prompt"] = template

    return messages


def pull_prompts_from_langsmith():
    print(f"Puxando prompt do LangSmith Hub: {HUB_PROMPT}")

    try:
        prompt_template = hub.pull(HUB_PROMPT)
    except Exception as e:
        print(f"❌ Erro ao puxar prompt do Hub: {e}")
        return None

    print(f"   ✓ Prompt carregado ({len(prompt_template.messages)} mensagens)")
    print(f"   ✓ Variáveis de entrada: {prompt_template.input_variables}")

    messages = extract_messages(prompt_template)

    if "system_prompt" not in messages:
        print("❌ O prompt puxado não possui uma mensagem de sistema")
        return None

    return {
        PROMPT_KEY: {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": messages["system_prompt"],
            "user_prompt": messages.get("user_prompt", "{bug_report}"),
            "version": "v1",
            "source": HUB_PROMPT,
            "pulled_at": date.today().isoformat(),
            "input_variables": list(prompt_template.input_variables),
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }


def main():
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    prompt_data = pull_prompts_from_langsmith()

    if prompt_data is None:
        return 1

    if not save_yaml(prompt_data, OUTPUT_FILE):
        return 1

    print(f"\n✅ Prompt salvo em: {OUTPUT_FILE}")
    print("\nPróximo passo: otimizar o prompt em prompts/bug_to_user_story_v2.yml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
