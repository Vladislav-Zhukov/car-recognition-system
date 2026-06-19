import torch

from ml.core.config import DEVICE, FAIR_EFFICIENTNET_MODEL_PATH, IMAGE_SIZE, MODEL_DIR
from ml.models.efficientnet import create_model

ONNX_MODEL_PATH = MODEL_DIR / "fair_efficientnet_100.onnx"


def main():
    checkpoint = torch.load(FAIR_EFFICIENTNET_MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = create_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to: {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()