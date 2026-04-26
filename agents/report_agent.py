def build_report(intent, plan, eval_result, improvement):
    lines = [
        intent["message"],
        plan["message"],
    ]
    if eval_result:
        lines.append(eval_result["message"])
    if improvement:
        lines.append(improvement["message"])
    return {
        "agent": "Report Agent",
        "summary": "\n".join(lines),
    }
