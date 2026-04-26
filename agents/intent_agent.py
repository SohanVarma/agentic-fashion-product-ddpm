def parse_intent(prompt: str, class_names):
    prompt_l = (prompt or "").lower()
    selected = None
    for cls in class_names:
        if cls.lower() in prompt_l:
            selected = cls
            break
    aliases = {
        "shoe": "Footwear", "shoes": "Footwear", "sneaker": "Footwear", "sandals": "Footwear",
        "shirt": "Apparel", "dress": "Apparel", "clothes": "Apparel", "clothing": "Apparel",
        "bag": "Accessories", "watch": "Accessories", "belt": "Accessories",
        "makeup": "Personal Care", "perfume": "Personal Care",
    }
    if selected is None:
        for word, mapped in aliases.items():
            if word in prompt_l and mapped in class_names:
                selected = mapped
                break
    return {
        "agent": "Intent Agent",
        "detected_class": selected or class_names[0],
        "message": f"Detected target category: {selected or class_names[0]}",
    }
