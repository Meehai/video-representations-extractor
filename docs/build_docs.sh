#!/usr/bin/env bash
# docs/build_docs.sh -- autogenerate the docs site with pdoc: themed (One Dark), self-contained, no
# sphinx/config files. Ported from github.com/Meehai/microecs (.build_docs.sh). Differences vs the
# microecs original, all driven by VRE's shape:
#   * TWO API packages are documented: vre/ (the engine) and vre_repository/ (the model zoo).
#   * The hand-written prose docs under docs/source/{basics,examples}/*.md are kept ON DISK AS-IS
#     and rendered ALONGSIDE the API: each becomes its own page under a synthetic "guide" package
#     (its docstring = the markdown), so the prose and the generated API share one themed sidebar.
#   * README.md is the home page (guide.html); index.html redirects to it.
#   * Vendored / third-party subtrees (*_impl, ultralytics, resources) and any module that fails to
#     import are excluded -- we document VRE's own API surface, not ported model internals.
#
#   bash docs/build_docs.sh            # build into ./public, then print a file:// link
#   bash docs/build_docs.sh public     # build into ./public  (GitLab Pages serves this dir in CI)
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root (this script lives in docs/)
OUT="${1:-public}"

export PYTHONPATH="$PWD:$PWD/vre-video:${PYTHONPATH:-}"

# one pure-python build dependency; auto-install so a fresh checkout just works
python -c "import pdoc" 2>/dev/null || { echo "[docs] installing pdoc..."; pip install --quiet pdoc; }

# --- embedded theme + template -> temp dir (pdoc prefers these files over its built-in defaults) ---
TMPL="$(mktemp -d)"
trap 'rm -rf "$TMPL"' EXIT

cat > "$TMPL/theme.css" <<'THEME_CSS'
:root {
  --pdoc-background: #282C34;
}
.pdoc {
  --text: white;
  --muted: whitesmoke;
  --link: var(--lightblue);
  --link-hover: white;
  --active: #555;
  --code: #232627;
  --accent: #232627;
  /* pdoc vars that onedark's theme.css omits -- map onto the palette so nothing falls back to plain text */
  --annotation: var(--orange);
  --def: var(--red);
  --name: var(--purple);
  --nav-hover: var(--dark);
  /* Actual theme colors */
  --lightblue: #61AFEF;
  --red: #E06C75;
  --green: #98C379;
  --dark: #282C34;
  --orange: #E5C07B;
  --purple: #B392F0;
  --blue: #9ECBFF;
  --silver: #ABB2BF;
}
THEME_CSS

cat > "$TMPL/custom.css" <<'CUSTOM_CSS'
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
}

.pdoc .il {
  color: #131416;
}

nav.pdoc a.pdoc-button.module-list-button {
  display: none;
}

.pdoc .modulename a:hover {
  color: whitesmoke;
}

.pdoc .classattr {
  color: #fff;
}

.pdoc .docstring .pdoc-code {
  background-color: #232627;
}

.pdoc pre {
  width: fit-content;
}

.pdoc pre a {
  color: whitesmoke;
}

.pdoc pre a:hover {
  color: #676767;
}


/* Submodules and API Doc. */

.pdoc h2 {
  color: #fff;
  font-weight: 0;
  margin: 0.3em 0;
  padding: 0.2em 0;
  content: "Module Content";
}

/* header */
.pdoc h5 {
  color: white;
  font-style: italic;
}

/* Module name */

.pdoc .modulename {
  color: #fff;
  font-weight: 0;
}

.pdoc-code .ow {
  color: var(--red);
}

nav.pdoc a.function {
  color: var(--purple);
}

nav.pdoc a.function:hover {
  background-color: var(--dark);
}

nav.pdoc .function::after {
  content: "";
}

nav.pdoc .function::before {
  content: "fn ";
  color: var(--red);
}

nav.pdoc a.class {
  color: whitesmoke;
}

/* Before the class name. This adds "class 'className'". */
nav.pdoc .class::before {
  content: "class ";
  color: var(--red);
}

/* After the class name. This will remove `()` That comes after the class name. */
nav.pdoc .class::after {
  content: "";
}

nav.pdoc a.class:hover {
  color: #fff;
  background-color: var(--dark);
}

nav.pdoc a.variable {
  color: whitesmoke;
}

/* Adds "var varname" to class variables. */
nav.pdoc a.variable::before {
  content: "var ";
  color: #9d9d9d;
  color: var(--red);
}

nav.pdoc a.variable:hover {
  color: #fff;
  background-color: var(--dark);
}

/* nav modules */
nav.pdoc li a {
  color: var(--purple);
}

nav.pdoc li a:hover {
  background-color: var(--dark);
}

/* Assigned values */
.pdoc span.default_value {
  color: var(--lightblue);
}

.pdoc span.def {
  font-weight: normal;
  color: var(--red);
}

/*
Attributes
----------
*/
.pdoc h6#attributes {
  color: #fff;
  font-size: 25px;
}

/*
Example
-------
*/

.pdoc h6#example {
  color: #fff;
  font-size: 29px;
}

/*
Notes
-----
*/

.pdoc h6#notes {
  font-size: 27px;
  color: orange;
}

/*
Returns
-------
*/

.pdoc h6#returns {
  font-size: 27px;
  color: white;
}

/*
Parameters
----------
*/

.pdoc h6#parameters {
  font-size: 27px;
  color: white;
}

/*
Raises
------
*/

.pdoc h6#raises {
  font-size: 27px;
  color: rgb(218, 55, 55);
}

/* False, True, None */

.pdoc .kc {
  color: var(--lightblue);
}

.pdoc li {
  color: white;
}

/* Decorator module */

.pdoc .nd {
  color: var(--lightblue);
}

.pdoc b,
strong {
  color: rgba(255, 255, 255, 0.8);
  font-weight: normal;
}

/* colors for .. warning:: and .. note:: */

.pdoc em {
  color: orange;
  font-size: 23px;
}

/* Decorator color */

.pdoc div.decorator {
  color: var(--lightblue);
}

div.pdoc-code.codehilite {
  background-color: var(--code);
}

/* class / function names */

.pdoc span.name {
  color: var(--purple);
  font-weight: normal;
}

/* Inherited class name color */

.pdoc span.base {
  color: whitesmoke;
}

/* Before inherited members colors, i.e., "builtins." */

.pdoc .inherited dt,
.pdoc .inherited dt::before {
  color: var(--silver);
}


/* Commas that separates parameters "," color */
.pdoc .inherited dd:not(:last-child)::after {
  color: #fff;
}

/* Commas that separates parameters "," color */

.pdoc .inherited dd:not(:last-child)::after {
  color: #fff;
}

/* Contents, Submodules, API Documentations */
/* This also can be separated */
.pdoc h1,
.pdoc h2,
.pdoc h3 {
  font-weight: 300;
  margin: 0.3em 0;
  padding: 0.2em 0;
  color: white;
}

/* Top left nav button */
nav.pdoc .module-list-button {
  display: inline-flex;
  align-items: center;
  margin-bottom: 1rem;
  color: white;
  border-color: white;
}

nav.pdoc .module-list-button:hover {
  border-color: white;
  color: white;
}

.pdoc pdoc-alert pdoc-alert-warning .p {
  color: black;
}

.pdoc-code {
  color: white;
}

.pdoc .pdoc-alert-warning {
  color: black;
  background-color: #d5a142;
  border-color: black;
}

.pdoc .pdoc-alert-note {
  color: rgb(7, 19, 24);
  background-color: rgb(184, 231, 251) ;
  border-color: var(--dark);
}

/* The nav bar */

nav.pdoc {
  background-color: rgb(55, 58, 72);
}

.pdoc-code .k {
  color: var(--red);
}

/* === local overrides (vre) ======================================================== */
/* widen the content column (layout.css ships 54rem) by 1.5x so wide code examples render
   on a single line, without horizontal scrolling */
main, header { width: calc(81rem + var(--sidebar-width)); }
/* a Home link at the top of the sidebar, above search; styled like the "Modules"/"Contents" headings */
nav.pdoc a.home-button { display: inline-block; color: #fff; font-weight: 300; font-size: 2rem; margin: 0.3em 0; }
nav.pdoc a.home-button:hover { color: var(--lightblue); }
/* README-heading primitives stay code in the body, but render as plain text in the sidebar menu */
nav.pdoc code { font-family: inherit; background: none; color: inherit; padding: 0; border: 0; font-size: inherit; }
/* prose/README images (e.g. the logo) must never overflow the content column */
.pdoc .docstring img { max-width: 100%; height: auto; }
/* Take over the tree's horizontal layout. pdoc indents each row via per-row <a> padding
   (calc(--pad + --indent)) plus a negative root-ul margin -- that does NOT compose into a real tree
   (steps are uneven and rails land at the far-left nav edge, not between levels). So: cancel the
   negative margin, give every row a small uniform padding, and let the nested <ul> padding be the
   ONLY indentation. Each level then steps evenly (~0.6rem) and its rail lines up under its parent. */
nav.pdoc > div > ul.module-tree { margin-left: 0; }
nav.pdoc ul.module-tree li a { padding-left: 0.4rem; }
/* tree-view rails: each nested group draws a faint vertical line spanning exactly its submodules.
   Hovering a package lights its subtree's rail (and its ancestors'), so the children stand out. */
nav.pdoc ul.module-tree ul {
  padding-left: 0.6rem;
  border-left: 1px solid rgba(255, 255, 255, 0.18);
}
nav.pdoc ul.module-tree li:hover > ul { border-left-color: var(--lightblue); }
CUSTOM_CSS

cat > "$TMPL/syntax-highlighting.css" <<'SYNTAX_CSS'
pre {
  line-height: 125%;
}

td.linenos pre {
  color: #ff5555;
  background-color: #282a36;
  padding-left: 5px;
  padding-right: 5px;
}

span.linenos {
  color: #ff5555;
  background-color: #282a36;
  padding-left: 5px;
  padding-right: 5px;
}

td.linenos pre.special {
  color: #ff5555;
  background-color: #ffffc0;
  padding-left: 5px;
  padding-right: 5px;
}

span.linenos.special {
  color: #ff5555;
  background-color: #ffffc0;
  padding-left: 5px;
  padding-right: 5px;
}

/* This is for the source code docs */

.pdoc-code .hll {
  background-color: #282a36;
}

.pdoc-code {
  background: #282a36;
  color: #ff5555;
}

/* Comment */

.pdoc-code .c {
  color: #6a7aaa;
}

/* Error */
.pdoc-code .err {
  color: #ff5555;
}

/* Keyword */
.pdoc-code .k {
  color: #ff79c6;
}

/* Literal */
.pdoc-code .l {
  color: #ae81ff;
}

/* Name */
.pdoc-code .n {
  color: #f8f8f2;
}

/* Operator */
.pdoc-code {
  color: #ff79c6;
}

/* Punctuation */
.pdoc-code .p {
  color: #f8f8f2;
}

/* Comment.Hashbang */
.pdoc-code .ch {
  color: #6a7aaa;
}

/* Comment.Multiline */
.pdoc-code .cm {
  color: #6a7aaa;
}

/* Comment.Preproc */
.pdoc-code .cp {
  color: #6a7aaa;
}

/* Comment.PreprocFile */
.pdoc-code .cpf {
  color: #6a7aaa;
}

/* Comment.Single */
.pdoc-code .c1 {
  color: var(--silver);
}

/* Comment.Special */
.pdoc-code .cs {
  color: #6a7aaa;
}

/* Generic.Deleted */
.pdoc-code .gd {
  color: #6a7aaa;
}

/* Generic.Emph */
.pdoc-code .ge {
  font-style: italic;
}

/* Generic.Inserted */
.pdoc-code .gi {
  color: #a6e22e;
}

/* Genericutput */
.pdoc-code .go {
  color: whitesmoke;
}

/* Generic.Prompt */
.pdoc-code .gp {
  color: rgb(171, 138, 193);
  font-weight: bold;
}

/* Generic.Strong */
.pdoc-code .gs {
  font-weight: bold;
}

/* Generic.Subheading */
.pdoc-code .gu {
  color: #75715e;
}

/* Keyword.Constant */
.pdoc-code .kc {
  color: white;
}

/* Keyword.Declaration */
.pdoc-code .kd {
  color: rgb(171, 138, 193);
}

/* Keyword.Namespace */
.pdoc-code .kn {
  color: var(--red);
}

/* Keyword.Pseudo */
.pdoc-code .kp {
  color: rgb(171, 138, 193);
}

/* Keyword.Reserved */

.pdoc-code .kr {
  color: rgb(171, 138, 193);
}

/* Keyword.Type */

.pdoc-code .kt {
  color: #ff79c6;
}

.pdoc-code .o {
  color: var(--red);
}

/* Literal.Date */

.pdoc-code .ld {
  color: #e6db74;
}

/* Literal.Number */

.pdoc-code .m {
  color: #ae81ff;
}

/* Literal.String */

.pdoc-code .s {
  color: var(--lightblue);
}

/* Name.Attribute */

.pdoc-code .na {
  color: #a6e22e;
}

/* Name.Builtin */

.pdoc-code .nb {
  color: var(--lightblue);
}

/* Name.Class */

.pdoc-code .nc {
  color: var(--purple);
}

/* Name.Constant */

.pdoc-code .no {
  color: #ff79c6;
}

/* Name.Decorator */

.pdoc-code .nd {
  color: #8be9fd;
}

/* Name.Entity */

.pdoc-code .ni {
  color: #ff79c6;
}

/* Name.Exception */

.pdoc-code .ne {
  color: #8be9fd;
}

/* Name.Function */

.pdoc-code .nf {
  color: var(--purple);
}

/* Name.Label */

.pdoc-code .nl {
  color: #ed9d13;
}

/* Name.Namespace */

.pdoc-code .nn {
  color: #f8f8f2;
}

/* Namether */

.pdoc-code .nx {
  color: #a6e22e;
}

/* Name.Property */

.pdoc-code .py {
  color: #f8f8f2;
}

/* Name.Tag */

.pdoc-code .nt {
  color: #f92672;
}

/* Name.Variable */

.pdoc-code .nv {
  color: #f8f8f2;
}

/* Operator.Word */

.pdoc-code w {
  color: #ff79c6;
}

/* Text.Whitespace */

.pdoc-code .w {
  color: #f8f8f2;
}

/* Literal.Number.Bin */

.pdoc-code .mb {
  color: #ae81ff;
}

/* Literal.Number.Float */

.pdoc-code .mf {
  color: #ae81ff;
}

/* Literal.Number.Hex */
.pdoc-code .mh {
  color: var(--lightblue);
}

/* Literal.Number.Integer */
.pdoc-code .mi {
  color: var(--lightblue);
}

/* Literal.Numberct */
.pdoc-code .mo {
  color: #ae81ff;
}

/* Literal.String.Affix */
.pdoc-code .sa {
  color: var(--red);
}

/* Literal.String.Backtick */
.pdoc-code .sb {
  color: #e6db74;
}

/* Literal.String.Char */
.pdoc-code .sc {
  color: #e6db74;
}

/* Literal.String.Delimiter */
.pdoc-code .dl {
  color: #e6db74;
}

/* Literal.String.Doc */
.pdoc-code .sd {
  color: var(--blue);
}

/* Literal.String.Double */
.pdoc-code .s2 {
  color: var(--blue);
}

/* Literal.String.Escape */
.pdoc-code .se {
  color: var(--lightblue);
}

/* Literal.String.Heredoc */
.pdoc-code .sh {
  color: #e6db74;
}

/* Literal.String.Interpol. AKA f-strings brackets */
.pdoc-code .si {
  color: var(--lightblue);
}

/* Literal.Stringther */
.pdoc-code .sx {
  color: #e6db74;
}

/* Literal.String.Regex */
.pdoc-code .sr {
  color: var(--blue);
}

/* Literal.String.Single */
.pdoc-code .s1 {
  color: var(--blue);
}

span.linenos {
  color: rgb(59, 145, 226);
  background-color: var(--code);
}

/* Literal.String.Symbol */

.pdoc-code .ss {
  color: #e6db74;
}

/* Name.Builtin.Pseudo */
.pdoc-code .bp {
  color: whitesmoke;
}

/* Name.Function.Magic */
.pdoc-code .fm {
  color: var(--lightblue);
}

/* Name.Variable.Class */
.pdoc-code .vc {
  color: #bd93f9;
}

/* Name.Variable.Global */
.pdoc-code .vg {
  color: #f8f8f2;
}

/* Name.Variable.Instance */
.pdoc-code .vi {
  color: #ffffff;
}

/* Name.Variable.Magic */
.pdoc-code .vm {
  color: var(--lightblue);
}

/* Literal.Number.Integer.Long */
.pdoc-code .il {
  color: var(--lightblue);
}
SYNTAX_CSS

# sidebar: show every top-level module on EVERY page (a persistent "Modules" menu), not only on the
# package landing. We override pdoc's nav_submodules block and list every rendered module. The Home
# link points at guide.html (the README home page built below).
cat > "$TMPL/module.html.jinja2" <<'JINJA'
{% extends "default/module.html.jinja2" %}
{% block module_list_link %}
    <a class="home-button" href="{{ "../" * module.modulename.count(".") }}guide.html">Home</a>
{% endblock %}
{# sidebar: the hand-written prose under a "Documentation" heading (friendly titles, fixed order
   via the doc_nav global), then the generated API under "Modules". The guide.* synthetic package
   is never shown verbatim. #}
{% block nav_submodules %}
    {% if doc_nav %}
    <h2>Documentation</h2>
    <ul>
        {% for name, title in doc_nav %}
            <li>{{ (name, "") | link(text=title) }}</li>
        {% endfor %}
    </ul>
    {% endif %}
    <h2>Modules</h2>
    {# nested package tree: api_roots = top packages, api_children maps a module -> its submodules.
       `recursive` + loop(kids) walks the hierarchy, so vre.utils nests vre.utils.atomic_open, etc. #}
    <ul class="module-tree">
    {% for name in api_roots recursive %}
        <li>
            {%- set leaf = name.split(".")[-1] %}
            {%- if name in api_pages %}{{ (name, "") | link(text=leaf) }}{% else %}{{ leaf }}{% endif -%}
            {% set kids = api_children.get(name) %}
            {% if kids %}<ul>{{ loop(kids) }}</ul>{% endif %}
        </li>
    {% endfor %}
    </ul>
{% endblock %}
{# prose pages render their own markdown H1 already; drop pdoc's "guide.xxx" modulename heading and
   the View-Source buttons for the synthetic guide package, keep them for the real API modules. #}
{% block module_info %}
    <section class="module-info">
        {% if module.modulename == "guide" or module.modulename.startswith("guide.") %}
            {{ docstring(module) }}
        {% else %}
            {{ module_name() }}
            {{ docstring(module) }}
            {{ view_source_state(module) }}
            {{ view_source_button(module) }}
            {{ view_source_code(module) }}
        {% endif %}
    </section>
{% endblock %}
JINJA

# --- render --------------------------------------------------------------------------------------
rm -rf "$OUT"
python - "$OUT" "$TMPL" <<'PY'
import sys, os, subprocess, pathlib, re, tempfile, importlib, warnings
import pdoc, pdoc.render
warnings.simplefilter("ignore")

out  = pathlib.Path(sys.argv[1])
tmpl = pathlib.Path(sys.argv[2])
root = pathlib.Path(".").resolve()

# 1) API surface: every .py under vre/ and vre_repository/, MINUS vendored/third-party subtrees
#    (*_impl, ultralytics, resources). Any module that fails to import is also dropped -- we don't
#    want ported model internals or broken stand-alone scripts in the API docs.
VENDORED = re.compile(r".*(_impl|ultralytics|\.resources)(\.|$)")
def candidates(pkg):
    mods = []
    for py in sorted((root / pkg).rglob("*.py")):
        parts = py.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name and not VENDORED.match(name):
            mods.append(name)
    return mods

bad = []
for m in candidates("vre") + candidates("vre_repository"):
    try:
        importlib.import_module(m)
    except Exception:
        bad.append(m)

# pdoc expands a package spec to ALL its submodules, so curation has to happen via "!" excludes:
# the vendored subtrees + every non-importable module discovered above (regex, exact-escaped).
excludes  = [r"!.*(_impl|ultralytics)(\.|$).*", r"!.*\.resources(\.|$).*"]
excludes += ["!" + re.escape(m) for m in bad]

# 2) Prose pages: the hand-written docs/source/{basics,examples}/*.md, kept on disk AS-IS. Each is
#    copied into the docstring of a synthetic "guide.<name>" module so pdoc renders it as a themed
#    page in the same sidebar as the API. README.md becomes the guide package docstring = home page.
md_files = sorted((root / "docs/source/basics").glob("*.md")) + [root / "docs/source/examples/example.md"]
stem2mod = {p.stem: p.stem.replace("-", "_") for p in md_files}   # cli-tools.md -> cli_tools.html

# --- portable links -------------------------------------------------------------------------------
# Prose (README + docs/source/*.md) is authored with repo-relative links so it also reads correctly on
# GitLab's file view. The GENERATED site is a SEPARATE artifact (only API + prose pages, served from
# any host/subpath), so rewrite link TARGETS -- only the `](...)` markdown form, never fenced code --
# so every link resolves from the site everywhere. `depth` = how many dirs deep the page sits:
#   1. self-links to our own Pages site (https://<proj>.gitlab.io/<proj>/X) -> page-relative X
#   2. repo paths not shipped in the site (cfg/scripts/notebooks/source)     -> gitlab.com blob/tree URL
#   3. inter-prose foo.md cross-links                                        -> the in-site foo.html
#   4. external / in-site .html / copied assets (logo.png)                   -> left as-is
def _git(*a, default=""):
    try:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return default
_slug   = re.sub(r"^.*gitlab\.com[:/]", "", _git("remote", "get-url", "origin",
            default="git@gitlab.com:video-representations-extractor/video-representations-extractor.git")).removesuffix(".git")
_branch = os.environ.get("CI_DEFAULT_BRANCH") or _git("symbolic-ref", "--short", "HEAD", default="master") or "master"
PAGES   = [p.rstrip("/") + "/" for p in
           {f"https://{_slug.split('/')[0]}.gitlab.io/{_slug.split('/')[-1]}", os.environ.get("CI_PAGES_URL", "")} if p]
REPO    = f"https://gitlab.com/{_slug}/-/%s/{_branch}/"      # %s -> "blob" (file) or "tree" (dir)
_LINK   = re.compile(r"\]\(([^)]+)\)")
def fixlinks(text, depth):
    up = "../" * depth
    def repl(m):
        tgt = m.group(1).strip()
        for pre in PAGES:                                          # 1. our Pages site -> relative
            if tgt.startswith(pre):
                return f"]({up}{tgt[len(pre):] or 'index.html'})"
        if re.match(r"^(https?:|mailto:|#)", tgt):                 # 4. external / pure anchor
            return m.group(0)
        path, sep, frag = tgt.partition("#")
        clean = path[2:] if path.startswith("./") else path
        if clean.endswith(".html") or clean == "logo.png":        # 4. already in-site
            return m.group(0)
        stem = pathlib.PurePosixPath(clean).stem
        if clean.endswith(".md") and stem in stem2mod:             # 3. prose cross-link -> in-site html
            return f"]({stem2mod[stem]}.html{sep}{frag})"
        return f"]({REPO % ('tree' if clean.endswith('/') else 'blob')}{clean}{sep}{frag})"  # 2. repo url
    return _LINK.sub(repl, text)

guidedir = pathlib.Path(tempfile.mkdtemp())
(guidedir / "guide").mkdir()
toc = "\n".join(f"- [{p.stem.replace('-', ' ').title()}](guide/{stem2mod[p.stem]}.html)" for p in md_files)
home = fixlinks((root / "README.md").read_text(encoding="utf-8"), 0) + f"\n\n## Documentation\n\n{toc}\n"
(guidedir / "guide" / "__init__.py").write_text('"""' + home + '"""\n', encoding="utf-8")
guide_mods = ["guide"]
for p in md_files:
    body = fixlinks(p.read_text(encoding="utf-8"), 1)
    (guidedir / "guide" / f"{stem2mod[p.stem]}.py").write_text('"""' + body + '"""\n', encoding="utf-8")
    guide_mods.append(f"guide.{stem2mod[p.stem]}")
sys.path.insert(0, str(guidedir))

# 3) render everything into one themed site. doc_nav drives the sidebar "Documentation" section:
#    (module name, friendly title) in the order the prose files were collected (basics, then example).
pdoc.render.configure(template_directory=tmpl)
pdoc.render.env.globals["doc_nav"] = [
    (f"guide.{stem2mod[p.stem]}", p.stem.replace("-", " ").title()) for p in md_files
]

# API sidebar as a package tree (not a flat list): every module that pdoc will render, plus its
# ancestor packages, wired parent -> children so the template can nest them recursively.
api_modules = [m for m in candidates("vre") + candidates("vre_repository") if m not in bad]
nodes = set(api_modules)
for m in api_modules:
    parts = m.split(".")
    for i in range(1, len(parts)):
        nodes.add(".".join(parts[:i]))
api_children = {}
for n in sorted(nodes):
    if "." in n:
        api_children.setdefault(n.rsplit(".", 1)[0], []).append(n)
pdoc.render.env.globals["api_roots"] = sorted(n for n in nodes if "." not in n)
pdoc.render.env.globals["api_children"] = api_children
pdoc.render.env.globals["api_pages"] = set(api_modules)

pdoc.pdoc("vre", "vre_repository", *excludes, *guide_mods, output_directory=out)

# 4) home page = README (guide.html); make the site root redirect there, and copy the logo so the
#    README's ![logo](logo.png) resolves on the landing page.
(out / "index.html").write_text(
    '<!doctype html><meta charset="utf-8">'
    '<meta http-equiv="refresh" content="0; url=guide.html">'
    '<link rel="canonical" href="guide.html"><title>VRE documentation</title>'
    '<a href="guide.html">Continue to the VRE documentation</a>\n', encoding="utf-8")
logo = root / "logo.png"
if logo.exists():
    (out / "logo.png").write_bytes(logo.read_bytes())

print(f"[docs] API modules: {len(candidates('vre')) + len(candidates('vre_repository')) - len(bad)} "
      f"(excluded {len(bad)} non-importable + vendored) | prose pages: {len(md_files)}")
PY

echo "[docs] built $OUT/  ->  open file://$(cd "$OUT" && pwd)/index.html"
