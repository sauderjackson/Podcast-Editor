"""Transcribe a podcast video with WhisperX for editing in Premiere Pro.

Outputs to transcripts/<name>/:
  <name>.srt   import into Premiere as captions (File > Import)
  <name>.json  word-level timestamps, used to find cuts (filler words, pauses)
  <name>.txt   plain transcript for reading

Usage:
  python transcribe.py media/episode.mp4 [--model large-v3] [--speakers 2 --hf-token TOKEN]
"""
import argparse
import os
from pathlib import Path

import whisperx
from whisperx.utils import get_writer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--language", default="en")
    p.add_argument("--speakers", type=int, help="number of speakers; enables diarization")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    name = Path(args.video).stem
    out = Path("transcripts") / name
    out.mkdir(parents=True, exist_ok=True)

    audio = whisperx.load_audio(args.video)

    model = whisperx.load_model(args.model, device, compute_type=compute, language=args.language)
    result = model.transcribe(audio, batch_size=8)

    align_model, meta = whisperx.load_align_model(language_code=args.language, device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device)

    if args.speakers:
        from whisperx.diarize import DiarizationPipeline
        diarize = DiarizationPipeline(token=args.hf_token, device=device)
        segs = diarize(audio, min_speakers=args.speakers, max_speakers=args.speakers)
        result = whisperx.assign_word_speakers(segs, result)

    result["language"] = args.language
    opts = {"max_line_width": 42, "max_line_count": 2, "highlight_words": False}
    for fmt in ("srt", "json", "txt"):
        get_writer(fmt, str(out))(result, args.video, opts)
    print(f"Done: {out}/")


if __name__ == "__main__":
    main()
