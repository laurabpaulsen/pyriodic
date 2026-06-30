from pathlib import Path
from ._fetch import fetch_dataset


_DATASET_NAME = "sample"

#  registry
BASE_URL = "https://osf.io/download/pby59/"
FNAME = "sample_data.pkl"


def data_path(
    path=None,
    force_update=False,
    download=True,
):
    """Return the local path to the sample dataset."""

    if path is None:
        path = Path.home() / ".pyriodic" / "datasets"

    path.mkdir(parents=True, exist_ok=True)

    dataset_path = path / _DATASET_NAME / FNAME

    if not dataset_path.exists():
        if not download:
            raise FileNotFoundError(
                "Sample dataset not found. "
                "Call data_path(download=True) to download it."
            )

        fetch_dataset(
            url=BASE_URL,
            fname=FNAME,
            destination=dataset_path.parent,
            force_update=force_update
            )
    

    return dataset_path