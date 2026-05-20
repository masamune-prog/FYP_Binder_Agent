SYSTEM_PROMPT = """\
web_search(f"{target_name} antibody binding epitope")
"""

try:
    SYSTEM_PROMPT.format(target_name="EGFR")
except Exception as e:
    import traceback
    traceback.print_exc()
