import torch
import torch.nn.functional as F


def evaluate_images(classifier, images, target_idx, class_names, device):
    if classifier is None:
        return {
            "agent": "Evaluator Agent",
            "available": False,
            "avg_confidence": None,
            "match_rate": None,
            "message": "Classifier checkpoint not found, so evaluation was skipped.",
        }
    classifier.eval()
    # classifier was trained on normalized [-1,1], generated images are [0,1]
    x = images.to(device) * 2 - 1
    with torch.no_grad():
        probs = F.softmax(classifier(x), dim=1)
        pred = probs.argmax(1)
        conf_target = probs[:, target_idx]
    match_rate = (pred == target_idx).float().mean().item()
    avg_conf = conf_target.mean().item()
    return {
        "agent": "Evaluator Agent",
        "available": True,
        "avg_confidence": avg_conf,
        "match_rate": match_rate,
        "predictions": [class_names[i] for i in pred.cpu().tolist()],
        "message": f"Target confidence={avg_conf:.2%}, match rate={match_rate:.2%}.",
    }
