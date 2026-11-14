"""python -m app.ingest  -> ingest the corpus and print what was found.

Shows per-document chunk counts, each document's lifecycle as of today (or --as-of),
and the data-quality findings (superseded versions, broken links, missing documents...).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

from app.config import settings
from app.ingest.pipeline import chunk_document, compute_lifecycles, load_corpus, quality_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", type=date.fromisoformat, default=settings.today())
    ap.add_argument("--verbose", "-v", action="store_true", help="list every document")
    args = ap.parse_args()

    docs, extra = load_corpus(settings.corpus_dir)
    lcs = compute_lifecycles([d.meta for d in docs], args.as_of)
    chunks = {d.doc_id: chunk_document(d) for d in docs}
    formats = Counter(d.meta.format for d in docs)
    print(f"{len(docs)} documents, {sum(map(len, chunks.values()))} chunks "
          f"({', '.join(f'{n} {f}' for f, n in sorted(formats.items()))}); lifecycle as of {args.as_of}")
    if args.verbose:
        for d in sorted(docs, key=lambda d: d.doc_id):
            print(f"  {d.doc_id:<32} {d.meta.format:<5} {len(chunks[d.doc_id]):>3} chunks  {lcs[d.doc_id].state}")
    print("\nData-quality findings:")
    for line in quality_report(docs, lcs, extra):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
