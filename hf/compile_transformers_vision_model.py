from modeling_dinov2_with_registers import Dinov2WithRegistersModel
from transformers import AutoModel
import fire
import torch


def dump_model(
    model_name: str = "facebook/dinov2-with-registers-base",
    output_path: str = "dinov2-with-registers-base.pt",
):
    """
    Converts a HuggingFace Vision Transformer model to a PyTorch JIT model.

    :param model_name: Any Hugging Face vision model, or absolute path to a local model.
    :param output_path: Location where the output PyTorch archive will be written.
    """

    if "dinov2-with-registers" in model_name:
        # Currently we must vendor this model due to bugs in the committed copy;
        # see https://github.com/huggingface/transformers/pull/35411
        model = Dinov2WithRegistersModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    # Set this so that we receive no conditional model control flow
    # and output a tuple instead of a dictionary.
    model.config.torchscript = True

    fn = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    fn.save(output_path)


if __name__ == "__main__":
    fire.Fire(dump_model)
