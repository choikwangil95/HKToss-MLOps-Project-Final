from optimum.onnxruntime import ORTModelForSeq2SeqLM
from tokenizers import Tokenizer
from pathlib import Path
import onnxruntime as ort


def get_summarize_model():
    """
    ONNX 요약 모델과 토크나이저 로딩
    """
    model_dir = (
        Path(__file__).resolve().parent.parent / "models" / "kobart_summary_onnx"
    )

    model = ORTModelForSeq2SeqLM.from_pretrained(
        str(model_dir), local_files_only=True  # ✅ str로!
    )

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    return model, tokenizer


def get_ner_tokenizer():
    """
    ONNX NER 모델과 토크나이저 로딩
    """
    model_dir = Path(__file__).resolve().parent.parent / "models" / "ner_onnx"

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "model.onnx"))

    return tokenizer, session
