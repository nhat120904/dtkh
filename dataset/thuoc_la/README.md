# Thước lá training images

The `rulers2023_*.jpg` files are square crops around ruler corner annotations from the Rulers2023 `real-train` set. Each image is resized to 384×384; source IDs remain in the filenames. Source: Dalius Matuzevicius, *Rulers2023: An Annotated Dataset of Synthetic and Real Images for Ruler Detection Using Deep Learning*, CC BY 4.0, https://doi.org/10.5281/zenodo.10276322.

The `openverse_*.jpg` files are manually screened real metal/steel rule photographs. Individual authors, licenses, and landing pages are recorded in `../CREDITS.csv`.

Rulers2023 contains several straight-edge ruler materials, including metal/steel examples. The app labels the class `Thước lá`; the model may also respond to visually similar straight rulers.
