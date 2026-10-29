"""
Evaluación de Whisper (inferencia, sin fine-tuning) sobre el test de quechua de Puno
y scoring con WER/CER usando la MISMA normalización para cualquier modelo.

Uso (PowerShell):

  pip install transformers torch librosa jiwer pandas tqdm

  # 1) Transcribir con Whisper (una corrida por condición de idioma: es, fi, auto)
  # (el TSV del repo trae rutas de Docker /workspace/data/...; --remap_* las convierte a locales)
  python whisper_eval.py transcribe --test_file results\\test_eval.tsv --remap_from /workspace/data --remap_to C:\\ruta\\local\\data --lang es   --out preds_whisper_es.csv
  python whisper_eval.py transcribe --test_file results\\test_eval.tsv --remap_from /workspace/data --remap_to C:\\ruta\\local\\data --lang fi   --out preds_whisper_fi.csv
  python whisper_eval.py transcribe --test_file results\\test_eval.tsv --remap_from /workspace/data --remap_to C:\\ruta\\local\\data --lang auto --out preds_whisper_auto.csv

  # Prueba rápida (10 clips)
  python whisper_eval.py transcribe --test_file results\\test_small.tsv --remap_from /workspace/data --remap_to C:\\ruta\\local\\data --lang es --out prueba.csv

  # 2) Calcular WER/CER de cualquier CSV con columnas id, ref, hyp
  #    (sirve igual para las predicciones de wav2vec2/XLS-R del repo de Huaman)
  python whisper_eval.py score preds_whisper_es.csv preds_whisper_fi.csv preds_whisper_auto.csv preds_wav2vec2.csv
"""
import argparse
import re
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd


def normalize(text: str) -> str:
    """Misma normalización para referencias y salidas de TODOS los modelos.
    Minúsculas, sin puntuación. Se conserva el apóstrofo porque en quechua
    marca consonantes glotalizadas (p', t', k', ch', q')."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = t.replace("|", " ")                # el repo separa palabras con '|'
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    t = re.sub(r"[^\w\s']", " ", t)       # quita puntuación, conserva '
    t = re.sub(r"\s+", " ", t).strip()
    return t


def transcribe(args):
    import librosa
    import torch
    from tqdm import tqdm
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Dispositivo: {device} | Modelo: {args.model}")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    sep = "\t" if args.test_file.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(args.test_file, sep=sep, quoting=3 if sep == "\t" else 0)
    if args.limit:
        df = df.head(args.limit)
    print(f"Audios a transcribir: {len(df)}")

    gen_kwargs = {"task": "transcribe", "num_beams": args.beams, "max_new_tokens": 200}
    if args.lang != "auto":
        gen_kwargs["language"] = args.lang  # "es" o "fi"; en auto el modelo detecta el idioma

    rows = []
    done = set()
    if Path(args.out).exists():              # reanudar si la corrida se interrumpió
        prev = pd.read_csv(args.out).fillna("")
        rows = prev.to_dict("records")
        done = set(prev["id"])
        print(f"Reanudando: {len(done)} clips ya transcritos en {args.out}")
    df = df[~df[args.audio_col].isin(done)]

    for n, start in enumerate(tqdm(range(0, len(df), args.batch_size))):
        batch = df.iloc[start:start + args.batch_size]
        audios = []
        for p in batch[args.audio_col]:
            p = str(p)
            if args.remap_from:                      # ruta de Docker -> ruta local
                p = p.replace(args.remap_from, args.remap_to, 1)
            full = Path(p) if Path(p).is_absolute() else Path(args.audio_dir) / p
            wav, _ = librosa.load(str(full), sr=16000, mono=True)
            audios.append(wav)
        feats = processor(audios, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(feats, **gen_kwargs)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for (_, r), h in zip(batch.iterrows(), hyps):
            rows.append({"id": r[args.audio_col], "ref": r[args.text_col], "hyp": h})
        if n % 10 == 9:                      # guardado parcial cada 10 lotes
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en {args.out}")
    report([args.out])


def report(files):
    results = []
    for f in files:
        df = pd.read_csv(f).fillna("")
        refs = df["ref"].map(normalize)
        hyps = df["hyp"].map(normalize)
        keep = refs.str.len() > 0          # jiwer falla con referencias vacías
        refs, hyps = refs[keep].tolist(), hyps[keep].tolist()
        results.append({
            "archivo": Path(f).name,
            "n": len(refs),
            "WER %": round(100 * jiwer.wer(refs, hyps), 2),
            "CER %": round(100 * jiwer.cer(refs, hyps), 2),
        })
    print("\n" + pd.DataFrame(results).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe")
    t.add_argument("--test_file", required=True, help="TSV/CSV del test con transcripción validada")
    t.add_argument("--audio_dir", default=".", help="Carpeta base si las rutas del TSV son relativas")
    t.add_argument("--remap_from", default="", help="Prefijo de ruta a reemplazar (ej. /workspace/data)")
    t.add_argument("--remap_to", default="", help="Prefijo local (ej. C:\\asr\\implementacion\\asr-puno-quechua-main\\data)")
    t.add_argument("--audio_col", default="path")
    t.add_argument("--text_col", default="transcription")
    t.add_argument("--model", default="openai/whisper-small")
    t.add_argument("--lang", default="es", choices=["es", "fi", "auto"])
    t.add_argument("--beams", type=int, default=1)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--out", required=True)
    t.set_defaults(func=transcribe)

    s = sub.add_parser("score")
    s.add_argument("files", nargs="+", help="CSVs con columnas id, ref, hyp")
    s.set_defaults(func=lambda a: report(a.files))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()