import os
import nbformat
import base64

from pathlib import Path

TUTORIALS_DIR = Path("docs_src/tutorials")
STATIC_IMG_DIR = Path("docs_src/_static/tutorials")
OUTPUT_RST = TUTORIALS_DIR / "index.rst"
MANIFEST_RST = TUTORIALS_DIR / "_image_manifest.rst"

intro_text = """\
Tutorials
=========

These tutorials walk you through using **Pyriodic** for circular data analysis, from getting started to phase diagnostics and circular statistics.

.. raw:: html

    <div class="row">
"""

card_template = """\
      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="{notebook_html}" class="stretched-link"></a>
          <img src="../_images/{image_file}" class="card-img-top" alt="{alt_text}">
          <div class="card-body">
            <h5 class="card-title">{title}</h5>
            <p class="card-text">{description}</p>
          </div>
        </div>
      </div>
"""

def extract_first_image(nb):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]
                return base64.b64decode(image_data)
    return None

def extract_title_and_desc(nb):
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines = cell.source.strip().splitlines()
            title = lines[0].strip().strip("# ") if lines else "Untitled"
            desc = lines[1].strip() if len(lines) > 1 else ""
            return title, desc
    return "Untitled", ""


def write_image_manifest(image_filenames, path):
    with open(path, "w") as f:
        f.write(".. This file is auto-generated to force Sphinx to include tutorial images.\n\n")
        f.write(".. raw:: html\n\n")
        f.write("   <div style='display:none'>\n\n")

        for image_file in image_filenames:
            f.write(f"   .. image:: /_images/{image_file}\n")
            f.write("      :alt: preload\n\n")

        f.write("   </div>\n")



def main():
    os.makedirs(STATIC_IMG_DIR, exist_ok=True)
    entries = []
    image_files_used = []

    for fname in sorted(TUTORIALS_DIR.glob("*.ipynb")):
        if fname.name.startswith("index"):
            continue

        nb = nbformat.read(fname, as_version=4)
        base = fname.stem
        title, desc = extract_title_and_desc(nb)

        # Image extraction
        image_file = f"{base}.png"
        image_data = extract_first_image(nb)
        image_path = STATIC_IMG_DIR / image_file

        if image_data:
            with open(image_path, "wb") as f:
                f.write(image_data)
                print(f"✅ Wrote image: {image_path}")
        else:
            image_file = "default.png"  # fallback
        image_files_used.append(image_file)

        entries.append(
            card_template.format(
                notebook_html=f"{base}.html",
                image_file=image_file,
                alt_text=title,
                title=title,
                description=desc or "See notebook for details.",
            )
        )

    # Write tutorials/index.rst
    notebook_stems = [f.stem for f in sorted(TUTORIALS_DIR.glob("*.ipynb")) if not f.name.startswith("index")]

    with open(OUTPUT_RST, "w") as f:
        f.write(intro_text + "\n".join(entries) + "\n    </div>\n\n")

        # Force Sphinx to copy images
        f.write(".. include:: _image_manifest.rst\n\n")

        # Hidden toctree for notebook indexing
        f.write(".. toctree::\n   :hidden:\n\n")
        for stem in notebook_stems:
            f.write(f"   {stem}\n")



    # Write _image_manifest.rst to force image inclusion
    write_image_manifest(image_files_used, MANIFEST_RST)

    print(f"\n✅ Generated {OUTPUT_RST} with {len(entries)} cards.")
    print(f"✅ Generated image manifest: {MANIFEST_RST}")

if __name__ == "__main__":
    main()
