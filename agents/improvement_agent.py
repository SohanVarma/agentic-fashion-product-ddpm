def should_regenerate(eval_result, plan, threshold=0.45):
    if not eval_result.get("available"):
        return {
            "agent": "Improvement Agent",
            "retry": False,
            "plan": plan,
            "message": "Evaluation unavailable; keeping first generation.",
        }
    if eval_result["avg_confidence"] is not None and eval_result["avg_confidence"] < threshold:
        new_plan = dict(plan)
        new_plan["guidance_scale"] = min(float(plan["guidance_scale"]) + 1.0, 6.0)
        return {
            "agent": "Improvement Agent",
            "retry": True,
            "plan": new_plan,
            "message": f"Confidence below {threshold:.0%}; regenerating with guidance={new_plan['guidance_scale']}.",
        }
    return {
        "agent": "Improvement Agent",
        "retry": False,
        "plan": plan,
        "message": "Quality acceptable; no regeneration needed.",
    }
