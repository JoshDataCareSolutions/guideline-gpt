# documents/

Drop your PDF guideline files in this folder, then build the index:

```bash
guideline-gpt ingest
```

The system is **corpus-agnostic** — any folder of PDFs works. The default
showcase corpus is public clinical guidelines (respiratory and critical care),
which you can fetch with:

```bash
python scripts/download_corpus.py
```

> **Note:** PDFs in this folder are git-ignored (only this README is tracked).
> Document sources, URLs, and license notes for the default corpus are recorded
> here by the download script.

---

## ⚠️ Disclaimer

This project is **not a medical device** and is **not for clinical
decision-making**. It operates only on public documents and performs no PHI
handling. Always consult primary sources and qualified professionals.
