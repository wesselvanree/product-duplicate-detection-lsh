from pathlib import Path


class Paths:
    repo_root = Path(__file__).parent.parent.parent.parent

    data = repo_root / "data"
