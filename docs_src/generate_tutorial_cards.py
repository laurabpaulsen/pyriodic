import os
import nbformat
import base64

from pathlib import Path

TUTORIALS_DIR = Path("docs_src/tutorials")
STATIC_IMG_DIR = Path("docs_src/_static/tutorials")
OUTPUT_RST = TUTORIALS_DIR / "index.rst"

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
          <img src="_static/tutorials/{image_file}" class="card-img-top" alt="{alt_text}">
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

def main():
    os.makedirs(STATIC_IMG_DIR, exist_ok=True)
    entries = []

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
        else:
            image_file = "default.png"  # fallback if no image found

        entries.append(
            card_template.format(
                notebook_html=f"{base}.html",
                image_file=image_file,
                alt_text=title,
                title=title,
                description=desc or "See notebook for details.",
            )
        )

    with open(OUTPUT_RST, "w") as f:
        f.write(intro_text + "\n".join(entries) + "\n    </div>\n")

    print(f"✅ Generated {OUTPUT_RST} with {len(entries)} cards.")

if __name__ == "__main__":
    main()
