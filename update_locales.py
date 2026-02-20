import os
import re

locales_dir = "spectrue_core/agents/locales/"

def update_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Pattern 1: Remove "- expert_summary: STRUCTURED analysis: Evidence, Gaps, Verdict, Style."
    content = re.sub(
        r'\s*-\s*expert_summary:\s*STRUCTURED analysis:.*?$',
        r'',
        content,
        flags=re.MULTILINE
    )

    # Pattern 2: the "Generate two versions" and "expert_summary: STRUCTURED analysis" text
    content = re.sub(
        r'Generate two versions:\n\s*1\.\s*\*\*simple_summary\*\*.*?(\n|.*?)\s*2\.\s*\*\*expert_summary\*\*:.*?STYLE_LABEL.*?$',
        r'Generate **simple_summary**: 1-3 concise bullet points for a general audience. No jargon.',
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    # A more robust regex for Pattern 2 (because locales have translated strings):
    # Basically look for "Generate two versions:" (translated) 
    # and replace the whole block until "ABSOLUTE RULES"
    content = re.sub(
        r'(\n\s*)(Generate two versions:|Створіть дві версії:|Создайте две версии:|生成两版本:|Generate|Cree dos versiones:|Erstellen Sie zwei Versionen:|Générer deux versions:|2つのバージョンを生成してください:).*?(?=\n\s*## )',
        r'\1Generate **simple_summary**: 1-3 concise bullet points for a general audience. No jargon.',
        content,
        flags=re.DOTALL
    )

    with open(path, 'w') as f:
        f.write(content)

for fn in os.listdir(locales_dir):
    if fn.endswith('.yml'):
        update_file(os.path.join(locales_dir, fn))

print("Updated locales")
