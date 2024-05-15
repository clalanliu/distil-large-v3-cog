# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from datasets import load_dataset
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        cache_dir = "model_cache"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "distil-whisper/distil-large-v3",
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to("cuda")
        self.processor = AutoProcessor.from_pretrained(
            "distil-whisper/distil-large-v3", cache_dir=cache_dir, local_files_only=True
        )

        self.model_en = AutoModelForSpeechSeq2Seq.from_pretrained(
            "distil-whisper/distil-medium.en",
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model_en.to("cuda")
        self.processor_en = AutoProcessor.from_pretrained(
            "distil-whisper/distil-medium.en",
            cache_dir=cache_dir,
            local_files_only=True,
        )

    def predict(
        self,
        audio: Path = Input(description="Input audio file"),
        model_name: str = Input(
            description="Choose a model.",
            choices=[
                "distil-whisper/distil-large-v3",
                "distil-whisper/distil-medium.en",
            ],
            default="distil-whisper/distil-large-v3",
        ),
        long_form_transcription: bool = Input(
            description="Enable chunked algorithm to transcribe long-form audio files.",
            default=False,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to output.", default=128
        ),
        batchsize: int = Input(
            description="Batchsize", default=128
        ),
    ) -> str:
        """Run a single prediction on the model"""
        model = (
            self.model
            if model_name == "distil-whisper/distil-large-v3"
            else self.model_en
        )
        processor = (
            self.processor
            if model_name == "distil-whisper/distil-large-v3"
            else self.processor_en
        )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=max_new_tokens,
            chunk_length_s=15 if long_form_transcription else 0,
            torch_dtype=torch.float16,
            device="cuda",
        )

        result = pipe(str(audio))
        return result["text"]
