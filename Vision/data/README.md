# Vision data pipeline

Code and settings for each stage of building the vision dataset and models, in the order they run. Implements Requirement Spec **Section 2.3 Vision Data and Models (VD)**.

| Folder | Stage | Spec |
|---|---|---|
| `preprocessing/` | Record clips as `.bag` (color and depth) and `.mp4` (for review) with pyrealsense2, name every clip and frame by fish ID and view, extract frames from the `.bag` files and drop blurry and near-duplicate frames | VD-10 to VD-37, Appendix C.1 |
| `annotating/` | Label frames in Label Studio with the mask-to-polygon tool, review the labels, and export them in both the segmentation and pose label formats | VD-40 to VD-65, Appendix C.2 |
| `training/` | Split the dataset into train, validation and test sets by fish ID (no fish in two sets), train the YOLO26 segmentation and pose models with MLflow tracking, and evaluate them, including on fish held out as unseen | VD-70 to VD-94, Appendix C.3 and C.10 |

The clips, frames, labels and model weights these stages read and write live on the team's dataset store, not in Git. `.gitignore` keeps recordings, video, images and weights out of the repository, so only code, configs and docs are committed here.
