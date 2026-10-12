
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

# Numerical values used only for the illustration.
Delta = 4.5
u = 0.3
p = 0.20

O = np.array([0.0, 0.0])
A = np.array([Delta, 0.0])
C = Delta * np.array([u, np.sqrt(2 * u - u**2)])
D = Delta * np.array([0.5, np.sqrt(3) / 2])

# Plotting window and density grid.
xmin, xmax = -2.5, 7.3
ymin, ymax = -2, 5.5
nx, ny = 1200, 980

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

def gaussian(mu):
    """Unnormalized N(mu, I_2) density; normalization is irrelevant here."""
    return np.exp(-0.5 * ((X - mu[0])**2 + (Y - mu[1])**2))

# Actual mixture densities.
fP = (1 - p) * gaussian(O) + p * gaussian(A)
fQ = (1 - p) * gaussian(O) + p * gaussian(C)

tP = fP / fP.max()
tQ = fQ / fQ.max()

def opacity(t):
    """Smooth density-to-opacity map: darker near modes, lighter in the tails."""
    base = t**0.56
    gate = np.clip((t - 0.0035) / 0.024, 0, 1)
    gate = gate * gate * (3 - 2 * gate)
    return 0.8 * base * gate

aP = opacity(tP)
aQ = opacity(tQ)

green = np.array([27, 158, 119]) / 255
purple = np.array([117, 112, 179]) / 255

# Symmetric alpha blending over a white background.
rgb = (
    (1 - aP[..., None] - aQ[..., None])
    + aP[..., None] * green
    + aQ[..., None] * purple
)
rgb = np.clip(rgb, 0, 1)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
})

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
fig.subplots_adjust(left=0.055, right=0.985, top=0.98, bottom=0.055)

# Mixture shading and true density contours.
ax.imshow(
    rgb,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    interpolation="bilinear",
    aspect="equal",
    zorder=0,
)

levels = [0.06, 0.12, 0.20, 0.34, 0.58]
ax.contour(
    X, Y, tP,
    levels=levels,
    colors=[green * 0.72],
    linewidths=0.8,
    alpha=0.43,
    zorder=1,
)
ax.contour(
    X, Y, tQ,
    levels=levels,
    colors=[purple * 0.72],
    linewidths=0.8,
    alpha=0.43,
    zorder=1,
)

# Coordinate axes.
axis_color = (0, 0, 0)
ax.annotate(
    "",
    xy=(xmax, 0),
    xytext=(xmin, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=1.0,
        color=axis_color,
        shrinkA=0,
        shrinkB=0,
    ),
    zorder=4,
)
ax.annotate(
    "",
    xy=(0, ymax),
    xytext=(0, ymin),
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=1.0,
        color=axis_color,
        shrinkA=0,
        shrinkB=0,
    ),
    zorder=4,
)
ax.text(xmax + 0.08, -0.05, r"$x$", ha="left", va="top",
        color=axis_color, fontsize=20)
ax.text(-0.07, ymax + 0.08, r"$y$", ha="right", va="bottom",
        color=axis_color, fontsize=20)

# Circular arc of radius Delta centered at Delta(1,0).
theta = np.linspace(np.pi, 2 * np.pi / 3, 250)
arc = A[:, None] + Delta * np.vstack((np.cos(theta), np.sin(theta)))
ax.plot(
    arc[0],
    arc[1],
    linestyle=(0, (3.0, 2.4)),
    linewidth=1.55,
    color=(0, 0, 0),
    zorder=5,
)

# Radius segments.
ax.plot(
    [A[0], C[0]], [A[1], C[1]],
    linewidth=1.05,
    color=(0, 0, 0),
    zorder=4,
)
ax.plot(
    [A[0], D[0]], [A[1], D[1]],
    linewidth=1.05,
    color=(0, 0, 0),
    zorder=4,
)

# Dimension |(0,0) - Delta(1,0)| = Delta.
ydim = -0.62
ax.plot([0, 0], [0, ydim], linewidth=0.8,
        color=(0, 0, 0), zorder=5)
ax.plot([Delta, Delta], [0, ydim], linewidth=0.8,
        color=(0, 0, 0), zorder=5)
ax.add_patch(FancyArrowPatch(
    (0, ydim),
    (Delta, ydim),
    arrowstyle="<->",
    mutation_scale=11,
    linewidth=1.0,
    color=(0, 0, 0),
    zorder=5,
))
ax.text(
    Delta / 2,
    ydim - 0.10,
    r"$\Delta=2/\sigma$",
    ha="center",
    va="top",
    fontsize=20,
    color=(0, 0, 0),
    zorder=6,
)

# Points.
points = np.vstack((O, A, C, D))
ax.scatter(points[:, 0], points[:, 1], s=32, c="black", zorder=7)

# Point labels, with no background boxes.
label_size = 20
ax.text(-0.18, -0.21, r"$(0,0)$",
        ha="right", va="top", fontsize=label_size, zorder=8)
ax.text(A[0] + 0.15, A[1] - 0.16, r"$\Delta(1,0)$",
        ha="left", va="top", fontsize=label_size, zorder=8)
ax.text(
    C[0] - 0.16,
    C[1] + 0.17,
    r"$\Delta(u,\sqrt{2u-u^2})$",
    ha="right",
    va="bottom",
    fontsize=label_size,
    zorder=8,
)
ax.text(
    D[0] + 0.14,
    D[1] + 0.18,
    r"$\Delta(1/2, \sqrt{3}/2)$",
    ha="left",
    va="bottom",
    fontsize=label_size,
    zorder=8,
)

ax.set_xlim(xmin, xmax + 0.15)
ax.set_ylim(ymin, ymax + 0.18)
ax.set_aspect("auto")
ax.axis("off")

# Boxed legend.
handles = [
    Patch(
        facecolor=green,
        edgecolor=green * 0.65,
        alpha=0.65,
        label=(
            r"$\Delta(1, 0) \times \mathrm{Ber}(p)+\mathcal{N}\ (0, I_2)$"
        ),
    ),
    Patch(
        facecolor=purple,
        edgecolor=purple * 0.65,
        alpha=0.65,
        label=(
            r"$\Delta(u, \sqrt{2u - u^2}) \times \mathrm{Ber}(p)+\mathcal{N}\ (0, I_2)$"
        ),
    ),
]
legend = ax.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(1.375, 0.775),
    ncol=1,
    frameon=True,
    fancybox=True,
    framealpha=1,
    borderpad=0.65,
    labelspacing=0.65,
    handlelength=1.25,
    handleheight=0.85,
    handletextpad=0.8,
    fontsize=15,
)
legend.get_frame().set_edgecolor((0, 0, 0))
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_facecolor("white")

output = "renyi_mixture_geometry.png"
fig.savefig(
    output,
    dpi=320,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.06,
)
plt.close(fig)

print(output)
