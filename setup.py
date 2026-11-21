"""setup.py for VRE. This allows to install using 'pip install -e .' on a dev machine (w/ venv/conda) after git clone"""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.20.3"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"

ROOT = Path(__file__).absolute().parent
with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Vendored packages: scan each repo dir under pkg/ for its packages (test/ trees are not installed).
vendored, vendored_dirs = [], {}
for repo in sorted(d for d in (ROOT / "pkg").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))):
    for name in find_packages(where=str(repo), exclude=["test", "test.*", "*.test", "*.test.*"]):
        vendored.append(name)
        vendored_dirs[name] = str((repo / name.replace(".", "/")).relative_to(ROOT))

def _filter_file(x: Path) -> bool:
    return (x.is_file() and x.suffix not in (".py", ".pyc", ".png", ".jpg", ".md")
            and x.name != ".gitignore" and "weights" not in x.parts)
glob_files = lambda x: list(Path(x).glob("**/*")) # pylint: disable=all

packages = find_packages() + vendored

def _build_package_data(roots: list[str]) -> dict[str, list[str]]:
    """non-py data files as package_data of their owning package: installed next to the code, where
    `Path(__file__).parent / ...` finds them (data_files would install them under sys.prefix)"""
    pkg_dirs = {p: Path(p.replace(".", "/")) for p in find_packages()}
    result: dict[str, list[str]] = {}
    for root in roots:
        for f in glob_files(root):
            if not _filter_file(f):
                continue
            owners = [(p, d) for p, d in pkg_dirs.items() if d in f.parents]
            assert owners, f"data file '{f}' is not inside any package; it won't be importable at runtime"
            pkg, pkg_dir = max(owners, key=lambda pd: len(pd[1].parts)) # deepest enclosing package
            result.setdefault(pkg, []).append(str(f.relative_to(pkg_dir)))
    return result

package_data = _build_package_data(["vre/", "vre_repository/"])

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=packages,
    package_dir=vendored_dirs,
    package_data=package_data,
    include_package_data=False,
    install_requires=[
        "loggez>=0.8.4",
        "numpy>=1.21.6",
        "Pillow>=11.3.0",
        "PyYAML==6.0.3",
        "tqdm==4.66.5",
        "graphviz==0.20.3",
        # torch is used just in vre_repository/ not in core vre
        "torch>=2.9.0,<3.0.0",
        "torchvision>=0.24.0,<1.0.0",
    ],
    extras_require={
        "cpu": [
            "torch==2.15.0+cpu; sys_platform != 'darwin'",
            "torchvision==0.29.2+cpu; sys_platform != 'darwin'",
        ],
        "docs": ["opdoc==0.2.3"],
    },
    dependency_links=[],
    license="MIT",
    python_requires=">=3.11",
    scripts=["cli/vre", "cli/vre_collage", "cli/vre_reader", "cli/vre_gpu_parallel", "cli/vre_dir_analysis"],
)
