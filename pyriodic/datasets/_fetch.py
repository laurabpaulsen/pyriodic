import pooch
from pathlib import Path



def fetch_dataset(
    destination: Path,
    url: str = None,
    fname: str = None,
    force_update: bool = False,
):
    """
    Download and prepare a dataset using pooch.

    Parameters
    ----------
    name : str
        Dataset identifier.
    destination : Path
        Where the dataset should live (already resolved by caller).
    force_update : bool
        If True, re-download even if cached.
    """


    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)


    pooch.retrieve(
        url=url,
        fname=fname,
        path=destination,
        known_hash=None,
        progressbar=True,
    )

