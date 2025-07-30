import os
import nbformat

TUTORIALS_DIR = "docs_src/tutorials"
STATIC_IMG_DIR = "_static/tutorials"
OUTPUT_RST = os.path.join(TUTORIALS_DIR, "index.rst")

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
          <img src="../{image_path}" class="card-img-top" alt="{alt_text}">
          <div class="card-body">
            <h5 class="card-title">{title}</h5>
            <p class="card-text">{description}</p>
          </div>
        </div>
      </div>
"""

def extract_notebook_title_and_description(nb_path):
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        cells = nb.cells
        for cell in cells:
            if cell.cell_type == "markdown":
                lines = cell.source.strip().splitlines()
                title = lines[0].strip().strip("# ") if lines else "Untitled"
                description = lines[1].strip() if len(lines) > 1 else ""
                return title, description
    except Exception:
        pass
    return "Untitled", ""

def main():
    entries = []
    for fname in sorted(os.listdir(TUTORIALS_DIR)):
        if fname.endswith(".ipynb") and not fname.startswith("index"):
            base = fname[:-6]  # remove .ipynb
            nb_path = os.path.join(TUTORIALS_DIR, fname)
            title, desc = extract_notebook_title_and_description(nb_path)
            img_path = f"{STATIC_IMG_DIR}/{base}.png"
            img_exists = os.path.exists(os.path.join("docs_src", img_path))
            entries.append(
                card_template.format(
                    notebook_html=f"{base}.html",
                    image_path=img_path if img_exists else "_static/default.png",
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
