# PneuInsight Studio

## Required asset files
Put these files in the same folder as `app.py`, or point `PNEUMO_ASSETS_DIR` to the folder that contains them:

- `checkpoint_infer.pt`
- `checkpoint_infer.json`
- `embeddings.npy`
- `rag_meta.csv`
- `knowledge_opt.jsonl`

## File list
- `app.py` — Streamlit GUI
- `model.py` — your ConvNeXtV2 + ECA + UNet++ + GeM model definition
- `rag_pipeline.py` — inference, image retrieval, text retrieval, payload building
- `report_generator.py` — LLM call, safety rewrite, fail-safe fallback
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional environment variables
```bash
# Windows PowerShell
$env:PNEUMO_ASSETS_DIR="D:\your\project\assets"
$env:DEEPSEEK_API_KEY="your_key_here"
$env:LLM_MODEL="deepseek-chat"
$env:LLM_API_URL="https://api.deepseek.com/chat/completions"
```

## Notes
- If `DEEPSEEK_API_KEY` is missing, the app still runs and uses a safe fallback template report.
- Similar case thumbnails are not shown, because your `rag_meta.csv` stores local paths that may not exist on another machine. The app still shows case IDs, similarity, and metadata.
- The app reads threshold, temperature, image size, normalization, and model kwargs directly from your exported checkpoint files.
