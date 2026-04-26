def plan_generation(intent_result, num_images, timesteps, guidance_scale, class_names):
    cls = intent_result["detected_class"]
    return {
        "agent": "Planner Agent",
        "class_name": cls,
        "class_index": class_names.index(cls),
        "num_images": int(num_images),
        "timesteps": int(timesteps),
        "guidance_scale": float(guidance_scale),
        "message": f"Planned {num_images} image(s), category={cls}, timesteps={timesteps}, guidance={guidance_scale}.",
    }
