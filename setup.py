"""setup.py -- note use setuptools==73.0.1; older versions fuck up the data files, newer versions include resources.

Import roots:
- `vre/` and `vre_repository/` are the app's own packages (imported as `import vre`, `import vre_repository`).
- each `pkg/<repo>/` is a vendored, in-tree copy of an external repo (pinned to an exact commit, no `.git`),
  auto-discovered and put on the import path so `import vre_video` / `import image_utils` just works.
  One layout: the package lives one level in (`pkg/vre-video/vre_video/`, `pkg/image_utils/image_utils/`).

`install_requires` / `extras_require` below are the single source of truth for deps (there are no
requirements*.txt files anymore). The vendored packages ship their own setup.py but we do NOT install
them as separate distributions -- they are just packages on our path.

Editable installs must take the PEP 660 path (it honours package_dir, i.e. the vendored pkg/ roots).
Old pip (e.g. 24.0) with setuptools+wheel installed falls back to legacy `setup.py develop`, which ignores
package_dir and leaves pkg/ unimportable. Fix: `pip install -U pip`. CI uses python:3.12 (no setuptools
preinstalled), so pip always builds via PEP 517/660 there.
"""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.19.0"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"

ROOT = Path(__file__).absolute().parent
with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_CORE = [
    "loggez>=0.8.4",
    "numpy>=1.21.6",
    "PyYAML==6.0.3",
    "tqdm==4.66.5",
    "overrides==7.7.0",
    "Pillow==11.3.0",
    "graphviz==0.20.3",
]

REQUIRED_REPOSITORY = [
    "torch==2.9.0",
    "torchvision==0.24.0",
    "opencv-python==4.14.0.94",
    "pycocotools==2.0.10",
    "timm==1.0.9",
    "diffusers==0.30.3",
]

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
    """Ship non-py data files (e.g. marigold's empty_text_embed.pkl, mask2former's *.json) as package_data
    keyed by their owning package, so they install *into* site-packages next to the code where
    `Path(__file__).parent / ...` can find them. data_files installs relative to sys.prefix instead, which
    flattens the paths and breaks those runtime lookups."""
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
    install_requires=REQUIRED_CORE,
    extras_require={
        "core": [],
        "repository": REQUIRED_REPOSITORY,
    },
    dependency_links=[],
    license="MIT",
    python_requires=">=3.10",
    scripts=["cli/vre", "cli/vre_collage", "cli/vre_reader", "cli/vre_gpu_parallel", "cli/vre_dir_analysis"],
)
