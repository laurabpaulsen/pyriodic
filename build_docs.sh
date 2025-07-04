pip install .
sphinx-build -M html docs_src docs
cp -r docs/html/* docs/ # Move built docs to root of /docs for GitHub Pages

#sphinx-build -M html docs docs/_build/
#cp -r docs/_build/html/* docs/