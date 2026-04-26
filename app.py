from pathlib import Path
import torch
import streamlit as st
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from model import ConditionalUNet
from classifier import FashionClassifier
from data_utils import FashionProductsDataset
from utils import plot_loss
from agents.intent_agent import parse_intent
from agents.planner_agent import plan_generation
from agents.generation_agent import generate_images
from agents.evaluator_agent import evaluate_images
from agents.improvement_agent import should_regenerate
from agents.report_agent import build_report

st.set_page_config(page_title="Agentic Fashion Product DDPM", page_icon="👗", layout="wide")
st.title("👗 Agentic Conditional Fashion Product Image Generator")
st.write("Generate RGB fashion product images using a class-conditional DDPM plus an agentic evaluation workflow.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DDPM_PATH = Path("checkpoints/conditional_fashion_product_ddpm.pt")
CLS_PATH = Path("checkpoints/fashion_product_classifier.pt")


def show_tensor_grid(images, title, nrow=4):
    if images is None or len(images) == 0:
        return
    grid = make_grid(images.detach().cpu(), nrow=nrow)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)


@st.cache_resource
def load_ddpm():
    if not DDPM_PATH.exists():
        return None, None
    ckpt = torch.load(DDPM_PATH, map_location=DEVICE)
    class_names = ckpt["class_names"]
    model = ConditionalUNet(num_classes=len(class_names), base_channels=ckpt.get("base_channels", 64)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@st.cache_resource
def load_classifier(num_classes):
    if not CLS_PATH.exists():
        return None
    ckpt = torch.load(CLS_PATH, map_location=DEVICE)
    model = FashionClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


model, ckpt = load_ddpm()

with st.sidebar:
    st.header("Project Controls")
    st.write(f"Device: `{DEVICE}`")
    if ckpt is None:
        st.warning("DDPM checkpoint not found. Train the model first.")
        class_names = ["Apparel", "Accessories", "Footwear", "Personal Care"]
        image_size = 64
        trained_timesteps = 100
    else:
        class_names = ckpt["class_names"]
        image_size = ckpt.get("image_size", 64)
        trained_timesteps = ckpt.get("timesteps", 100)
        st.success("DDPM checkpoint loaded")

    prompt = st.text_area("Describe your generation goal", "Generate 8 footwear product images")
    selected_class = st.selectbox("Or select category manually", class_names)
    num_images = st.slider("Number of images", 1, 16, 8)
    timesteps = st.slider("Sampling timesteps", 25, max(50, trained_timesteps), trained_timesteps)
    guidance = st.slider("Classifier-free guidance scale", 0.0, 6.0, 2.5, 0.5)
    use_prompt_class = st.checkbox("Let Intent Agent choose class from prompt", value=True)
    generate_btn = st.button("Generate")

st.subheader("What this project does")
st.write(
    "This project upgrades basic DDPM generation into a controlled fashion product generator. "
    "The model is trained on RGB product images and uses class labels so the user can request categories such as Apparel, Accessories, Footwear, or Personal Care."
)

if model is None:
    st.info("Run the commands below before using the generator:")
    st.code(
        "python download_dataset.py\n"
        "python train_classifier.py --epochs 5 --batch_size 64 --image_size 64 --max_images 12000\n"
        "python train_diffusion.py --epochs 10 --batch_size 32 --timesteps 100 --image_size 64 --max_images 12000\n"
        "python -m streamlit run app.py",
        language="bash",
    )
else:
    classifier = load_classifier(len(class_names))
    if generate_btn:
        if use_prompt_class:
            intent = parse_intent(prompt, class_names)
        else:
            intent = {"agent": "Intent Agent", "detected_class": selected_class, "message": f"Manual class selected: {selected_class}"}

        plan = plan_generation(intent, num_images, timesteps, guidance, class_names)

        logs = [intent["message"], plan["message"]]
        gen = generate_images(model, plan, image_size, DEVICE)
        logs.append(gen["message"])

        eval_result = evaluate_images(classifier, gen["images"], plan["class_index"], class_names, DEVICE)
        logs.append(eval_result["message"])

        improvement = should_regenerate(eval_result, plan)
        logs.append(improvement["message"])

        final_images = gen["images"]
        final_plan = plan
        if improvement["retry"]:
            final_plan = improvement["plan"]
            gen2 = generate_images(model, final_plan, image_size, DEVICE)
            final_images = gen2["images"]
            logs.append(gen2["message"])
            eval_result = evaluate_images(classifier, final_images, final_plan["class_index"], class_names, DEVICE)
            logs.append("After retry: " + eval_result["message"])

        report = build_report(intent, final_plan, eval_result, improvement)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Generated Images")
            show_tensor_grid(final_images, f"Generated: {final_plan['class_name']}", nrow=4)
        with right:
            st.subheader("Agent Workflow Log")
            for i, line in enumerate(logs, start=1):
                st.write(f"{i}. {line}")
            st.subheader("Final Report")
            st.text(report["summary"])

        st.subheader("Real vs Generated Comparison")
        try:
            ds = FashionProductsDataset(image_size=image_size, class_names=class_names, max_images=3000)
            real = ds.get_real_samples(final_plan["class_index"], count=min(num_images, 8))
            c1, c2 = st.columns(2)
            with c1:
                show_tensor_grid(real, f"Real {final_plan['class_name']} samples", nrow=4)
            with c2:
                show_tensor_grid(final_images[: min(num_images, 8)], f"Generated {final_plan['class_name']} samples", nrow=4)
        except Exception as exc:
            st.warning(f"Could not load real samples for comparison: {exc}")

st.subheader("Training Loss")
fig = plot_loss("outputs/training_loss.csv")
if fig:
    st.pyplot(fig)
else:
    st.info("Training loss graph will appear after running train_diffusion.py")
