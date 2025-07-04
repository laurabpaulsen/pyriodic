pip install .
sphinx-build -M html docs docs/_build/
cp -r docs/_build/html/* docs/