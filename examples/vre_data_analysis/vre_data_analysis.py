import sys
from vre.readers import MultiTaskDataset
from vre.representations import build_representations_from_cfg, Representation
from vre.representations.cv_representations import SemanticRepresentation
from vre.logger import vre_logger as logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import bs4
from PIL import Image
import seaborn as sns

def extract_pil_from_b64_image(base64_buf: str) -> Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_buf)))

def extract_b64_image_from_fig(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi)
    buffer.seek(0)
    base64_buf = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_buf

def extract_b64_imgsrc_from_fig(fig: plt.Figure) -> str:
    base64_buf = extract_b64_image_from_fig(fig)
    return f"""<img src="data:image/png;base64,{base64_buf}" alt="Sample Plot">"""

def save_html(html_imgs: list[str], description: str, out_path: str):
    html = bs4.BeautifulSoup(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VRE Dataset Analysis</title>
    </head>
    <body>
        <h1 id="description">Description</h1>
        <h1 id="plots">Plots</h1>
    </body>
    </html>""", features="lxml")
    html.find(id="description").insert_after(bs4.BeautifulSoup(description.replace("\n", "<br/>"), features="lxml"))
    for html_img in html_imgs[::-1]:
        html.find(id="plots").insert_after(bs4.BeautifulSoup(html_img, features="lxml"))
    open(out_path, "w").write(str(html))
    print(f"Written html at '{out_path}'")


def histogram_from_classification_task(reader: MultiTaskDataset, classif: SemanticRepresentation,
                                       n: int | None = None, mode: str = "sequential", **figkwargs) -> plt.Figure:
    fig = plt.Figure(**figkwargs)
    counts = np.zeros(len(classif.classes), dtype=np.uint64)
    ixs = np.arange(len(reader)) if mode == "sequential" else np.random.permutation(len(reader))
    ixs = ixs[0:n] if n is not None and n < len(reader) else ixs
    assert getattr(classif, "load_mode", "binary") == "binary", classif.load_mode
    for i in ixs:
        item = reader.get_one_item(i.item(), subset_tasks=[classif.name])
        data_cnts = item[0][classif.name].unique(return_counts=True)
        item_classes, item_counts = data_cnts[0].numpy().astype(int), data_cnts[1].numpy().astype(int)
        counts[item_classes] = counts[item_classes] + item_counts

    df = pd.DataFrame({"Labels": classif.classes, "Values": counts})
    df["Values"] = df["Values"] / df["Values"].sum()
    df = df.sort_values("Values", ascending=True)
    df = df[df["Values"] > 0.005]

    ax = fig.gca()
    sns.barplot(data=df, y="Labels", x="Values", palette="viridis", legend=True, ax=ax, width=1)

    # Adjust y-axis tick positions and spacing
    ax.set_title(classif.name, fontsize=14, fontweight='bold')
    ax.set_ylabel("Labels", fontsize=12)

    fig.set_size_inches(8, 2 if len(df) <= 2 else len(df) * 0.5)
    fig.gca().set_xlim(0, 1)
    fig.tight_layout()
    plt.close()
    return fig

def gaussian_from_statistics(reader: MultiTaskDataset, regression_task: Representation) -> plt.Figure:
    _, __, mean, std = [x.numpy() for x in reader.statistics[regression_task.name]]
    fig, ax = plt.subplots(1, n_ch := mean.shape[0], figsize=(10, 5))
    ax = [ax] if n_ch == 1 else ax
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    for i in range(n_ch):
        ax[i].plot(x[:, i], y[:, i])
    fig.suptitle(regression_task.name)
    fig.gca().set_xlabel("Mean")
    fig.gca().set_xlabel("Std")
    return fig

if __name__ == "__main__":
    data_path = sys.argv[1]
    cfg_path = sys.argv[2]
    representations = build_representations_from_cfg(cfg_path)
    print(representations)
    reader = MultiTaskDataset(data_path, task_names=list(representations),
                              task_types=representations, normalization="min_max")
    print(reader)

    imgsrcs = []
    for classif_task in reader.classification_tasks:
        fig = histogram_from_classification_task(reader, classif_task)
        imgsrcs.append(extract_b64_imgsrc_from_fig(fig))

    regression_tasks = [t for t in reader.tasks if t not in reader.classification_tasks]
    for regression_task in regression_tasks:
        fig = gaussian_from_statistics(reader, regression_task)
        imgsrcs.append(extract_b64_imgsrc_from_fig(fig))

    save_html(imgsrcs, str(reader), "plot.html")
