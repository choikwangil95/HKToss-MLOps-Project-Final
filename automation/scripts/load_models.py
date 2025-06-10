from tokenizers import Tokenizer
from pathlib import Path
import onnxruntime as ort


def get_summarize_model():
    """
    ONNX 요약 모델과 토크나이저 로딩
    """
    base_path = Path("../../automation/models/kobart_summary_onnx")

    encoder_sess = ort.InferenceSession(str(base_path / "encoder_model.onnx"))
    decoder_sess = ort.InferenceSession(str(base_path / "decoder_model.onnx"))
    tokenizer = Tokenizer.from_file(str(base_path / "tokenizer.json"))

    return encoder_sess, decoder_sess, tokenizer


def get_ner_tokenizer():
    """
    ONNX NER 모델과 토크나이저 로딩
    """
    model_dir = Path(__file__).resolve().parent.parent / "models" / "ner_onnx"

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "model.onnx"))

    return tokenizer, session
