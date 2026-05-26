"""Streamlit UI for the Image Similarity Sorter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from sorter import (
    APP_DIR,
    DEFAULT_THRESHOLD,
    DEVICE,
    MASTERS_DIR,
    OUTPUT_DIR,
    classify_images,
    encode_images,
    load_clip,
    load_masters,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@st.cache_resource
def cached_clip():
    return load_clip()


@st.cache_data
def cached_masters(_model, _processor, masters_dir: str):
    return load_masters(Path(masters_dir), _model, _processor)


def main() -> None:
    st.set_page_config(page_title="Image Similarity Sorter", layout="wide")
    st.title("Image Similarity Sorter")
    st.caption("CLIP-based visual matching against master reference images")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        st.write(f"**Device:** {DEVICE}")

        masters_dir = st.text_input("Masters folder", value=str(MASTERS_DIR))
        survey_dir = st.text_input(
            "Survey folder",
            value=r"D:\DETERIOROS TEXCOCO\RAMAL 5 SENTIDO 1\Derecha",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.50, max_value=0.99, value=DEFAULT_THRESHOLD, step=0.01,
            help="Images below this cosine similarity are sent to _no_damage",
        )
        copy_mode = st.checkbox("Copy files (uncheck to move)", value=True)

        limit = st.number_input(
            "Max images to scan (0 = all)", min_value=0, value=0, step=100,
            help="Useful for quick testing on a subset",
        )

        # Show master classes
        masters_path = Path(masters_dir)
        if masters_path.is_dir():
            st.divider()
            st.subheader("Master Classes")
            for d in sorted(masters_path.iterdir()):
                if d.is_dir():
                    n = len([f for f in d.iterdir() if f.suffix.lower() in EXTENSIONS])
                    st.write(f"**{d.name}**: {n} images")

    # ── Main area ─────────────────────────────────────────────────────────────
    tab_scan, tab_results, tab_masters = st.tabs(["Scan", "Results", "Masters Preview"])

    with tab_masters:
        st.subheader("Master Reference Images")
        masters_path = Path(masters_dir)
        if not masters_path.is_dir():
            st.warning("Masters folder not found")
        else:
            for cls_dir in sorted(masters_path.iterdir()):
                if not cls_dir.is_dir():
                    continue
                imgs = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in EXTENSIONS])
                if not imgs:
                    continue
                st.markdown(f"### {cls_dir.name} ({len(imgs)} images)")
                cols = st.columns(min(len(imgs), 6))
                for col, img_path in zip(cols, imgs):
                    with col:
                        st.image(str(img_path), caption=img_path.name, width="stretch")

    with tab_scan:
        st.subheader("Run Similarity Sort")

        survey_path = Path(survey_dir)
        if not survey_path.is_dir():
            st.warning("Survey folder not found")
            return

        survey_files = sorted([f for f in survey_path.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS])
        st.info(f"**{len(survey_files)}** images in survey folder")

        run = st.button("Run Sort", type="primary")

        if run:
            files_to_scan = survey_files if limit == 0 else survey_files[:limit]

            with st.spinner("Loading CLIP model..."):
                model, processor = cached_clip()

            with st.spinner("Encoding master images..."):
                class_embeds = cached_masters(model, processor, masters_dir)

            if not class_embeds:
                st.error("No master classes found!")
                return

            # Encode survey images with progress
            progress = st.progress(0, text="Encoding survey images...")
            batch_size = 64
            all_embeds = []
            for start in range(0, len(files_to_scan), batch_size):
                batch = files_to_scan[start : start + batch_size]
                pil_images = []
                for p in batch:
                    try:
                        pil_images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pil_images.append(Image.new("RGB", (224, 224)))
                inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(DEVICE)
                import torch
                with torch.no_grad():
                    out = model.get_image_features(**inputs)
                    embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
                    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu().numpy())
                pct = min((start + len(batch)) / len(files_to_scan), 1.0)
                progress.progress(pct, text=f"Encoded {min(start + len(batch), len(files_to_scan))}/{len(files_to_scan)}")

            import numpy as np
            survey_embeds = np.concatenate(all_embeds, axis=0)
            progress.empty()

            # Classify
            assignments = classify_images(survey_embeds, class_embeds, threshold)

            # Build results table
            rows = []
            for survey_file, (cls_name, sim) in zip(files_to_scan, assignments):
                rows.append({
                    "filename": survey_file.name,
                    "class": cls_name,
                    "similarity": round(sim, 4),
                    "path": str(survey_file),
                })

            df = pd.DataFrame(rows)
            st.session_state.sort_results = df
            st.session_state.sort_threshold = threshold
            st.session_state.sort_embeds = survey_embeds
            st.session_state.sort_class_embeds = class_embeds
            st.session_state.sort_files = files_to_scan

            # Summary
            from collections import Counter
            counts = Counter(r["class"] for r in rows)
            st.success(f"Sorted {len(rows)} images:")
            for cls, n in sorted(counts.items()):
                st.write(f"  **{cls}**: {n}")

        # Re-threshold without re-encoding
        if "sort_embeds" in st.session_state:
            st.divider()
            st.subheader("Adjust Threshold (instant)")
            new_thresh = st.slider(
                "Re-classify threshold",
                min_value=0.50, max_value=0.95,
                value=st.session_state.get("sort_threshold", DEFAULT_THRESHOLD),
                step=0.01, key="rethresh",
            )
            if st.button("Re-classify", key="reclassify"):
                assignments = classify_images(
                    st.session_state.sort_embeds,
                    st.session_state.sort_class_embeds,
                    new_thresh,
                )
                rows = []
                for survey_file, (cls_name, sim) in zip(st.session_state.sort_files, assignments):
                    rows.append({
                        "filename": survey_file.name,
                        "class": cls_name,
                        "similarity": round(sim, 4),
                        "path": str(survey_file),
                    })
                df = pd.DataFrame(rows)
                st.session_state.sort_results = df
                st.session_state.sort_threshold = new_thresh

                from collections import Counter
                counts = Counter(r["class"] for r in rows)
                st.success(f"Re-classified {len(rows)} images at threshold {new_thresh}:")
                for cls, n in sorted(counts.items()):
                    st.write(f"  **{cls}**: {n}")

        # Copy/move button
        if "sort_results" in st.session_state and not st.session_state.sort_results.empty:
            st.divider()
            if st.button("Copy sorted images to output folders", type="primary", key="do_copy"):
                import shutil
                df = st.session_state.sort_results
                out = Path(OUTPUT_DIR)
                out.mkdir(parents=True, exist_ok=True)
                for cls in df["class"].unique():
                    (out / cls).mkdir(exist_ok=True)

                prog = st.progress(0, text="Copying...")
                for i, row in df.iterrows():
                    src = Path(row["path"])
                    dest = out / row["class"] / row["filename"]
                    if copy_mode:
                        shutil.copy2(src, dest)
                    else:
                        shutil.move(str(src), str(dest))
                    prog.progress((i + 1) / len(df), text=f"Copied {i + 1}/{len(df)}")
                prog.empty()
                st.success(f"Done! Files in `{out}`")

    with tab_results:
        if "sort_results" not in st.session_state or st.session_state.sort_results.empty:
            st.info("Run a scan first.")
            return

        df = st.session_state.sort_results
        st.subheader(f"Results ({len(df)} images)")

        # Filter
        classes = ["All"] + sorted(df["class"].unique().tolist())
        show_cls = st.selectbox("Filter by class", classes)
        filtered = df if show_cls == "All" else df[df["class"] == show_cls]

        st.dataframe(
            filtered,
            hide_index=True,
            width="stretch",
            column_config={
                "similarity": st.column_config.NumberColumn(format="%.4f"),
                "path": None,
            },
        )

        # Gallery
        st.divider()
        st.subheader("Gallery")
        cols_per_row = st.select_slider("Columns", [2, 3, 4, 5, 6], value=4, key="gal_cols")
        page_size = cols_per_row * 4
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input("Page", 1, total_pages, 1, key="gal_page")
        page_items = filtered.iloc[(page - 1) * page_size : page * page_size]

        for chunk_start in range(0, len(page_items), cols_per_row):
            chunk = page_items.iloc[chunk_start : chunk_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    img_path = Path(row["path"])
                    if img_path.is_file():
                        st.image(str(img_path), width="stretch")
                    st.caption(f"{row['filename']}\n{row['class']} ({row['similarity']:.4f})")


if __name__ == "__main__":
    main()
