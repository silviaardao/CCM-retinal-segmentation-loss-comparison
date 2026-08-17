# Loss functions and evaluation metrics

This folder contains the objectives used to train the models and the metrics used to evaluate
their predictions. They are kept together because both work with the same five segmentation
classes: background, artery, vein, artery-vein overlap, and ambiguous vessel.

`losses/segmentation_losses.py` defines the ten loss configurations. `losses/structure_masks.py`
derives the crossover and branch masks needed by the CBAV adaptation. The metrics in
`metrics/segmentation_metrics.py` are shared by the main U-Net study and the nnU-Net
comparison.

## Baseline losses

- **Cross-entropy (`ce`)** compares the predicted class probability with the target class at
  every pixel.
- **Dice (`dice`)** measures overlap between the predicted and target regions. Background is
  excluded from the multiclass Dice objective.
- **CE + Dice (`ce_dice`)** combines unweighted cross-entropy with Dice.
- **Weighted CE + Dice (`weighted_ce_dice`)** uses class weights to reduce the dominance of
  background and give more importance to the vessel classes.
- **Focal + Dice (`focal_dice`)** uses focal loss to give more attention to pixels that are
  difficult to classify, together with Dice overlap.

Dice follows Milletari, Navab and Ahmadi (2016), *V-Net: Fully Convolutional Neural Networks
for Volumetric Medical Image Segmentation*, https://doi.org/10.1109/3DV.2016.79. Focal loss
follows Lin et al. (2017), *Focal Loss for Dense Object Detection*,
https://doi.org/10.1109/ICCV.2017.324.

## Structure-aware losses

### CF-Loss

CF-Loss adds penalties for clinically relevant vascular features to a cross-entropy base:

- `cf_v` adds vessel-density agreement;
- `cf_b` adds multiscale soft box-count agreement, related to fractal dimension;
- `cf_vb` adds both terms.

The implementation uses the source coefficients: vessel-density beta=1.0 and box-count
lambda=0.5. The three modes activate only the terms named above. In this project, the feature
terms are calculated for the artery and vein classes within the common five-class output.

Source: Zhou et al. (2024), *CF-Loss: Clinically-Relevant Feature Optimised Loss Function for
Retinal Multi-Class Vessel Segmentation and Vascular Feature Measurement*, Medical Image
Analysis, 93, 103098. https://doi.org/10.1016/j.media.2024.103098.

### CBAV-Loss adaptation

CBAV-Loss focuses on errors near artery-vein crossings and branch points. The source method
was developed for OCTA images. This project adapts it to fundus annotations, where a crossing
is stored as its own overlap class rather than as two simultaneous labels.

To preserve the meaning of a crossing, overlap pixels are included in both the artery and
vein masks before their skeletons are calculated. Crossover and branch masks are derived from
the ground-truth label during loss calculation; they are not separate prediction targets and
are not saved over the original colour masks. The objective uses the source settings
`lambda_cross=0.005`, `lambda_branch=0.005`, crossover radius 8, and branch radius 5.

Source: Zhang et al. (2023), *CBAV-Loss: Crossover and Branch Losses for Artery-Vein
Segmentation in OCTA Images*, Pattern Recognition and Computer Vision, 51–60.
https://doi.org/10.1007/978-981-99-8558-6_5.

### Soft-clDice adaptation

clDice compares vessel centre lines as well as their surrounding masks. It is useful for
tubular structures because an ordinary overlap score can remain high even when a vessel is
broken. The differentiable soft-clDice version can be used during training.

The project loss is weighted CE + Dice + `0.1 * soft-clDice`. Artery-vein overlap pixels are
included in both vessel trees. The CE anchor, the coefficient 0.1, and the five-class mapping
are project choices rather than the original paper's reference configuration.

Source: Shit et al. (2021), *clDice—A Novel Topology-Preserving Loss Function for Tubular
Structure Segmentation*, CVPR, 16555–16564.
https://doi.org/10.1109/CVPR46437.2021.01629.

## Evaluation metrics

- **Per-class Dice** measures pixel overlap for background, artery, vein, overlap, and
  ambiguous vessel. The main AV summary is the mean of artery and vein Dice.
- **clDice** measures centreline agreement for the artery and vein trees. This is an
  evaluation metric as well as the basis of one training objective, but the evaluation uses
  hard predicted labels rather than the differentiable training approximation.
- **Betti-0 error** is the absolute difference between the number of connected components in
  a prediction and its ground truth. A lower value indicates closer agreement in
  fragmentation.
- **Vessel-density error** is the absolute difference between predicted and target artery or
  vein area fractions.
- **Pixel accuracy and the confusion matrix** provide an overall view of class assignments.

Dice, clDice, and Betti-0 are calculated per image. If a class is absent from an image's
ground truth, that image is omitted from the mean for that class instead of being treated as
a perfect or failed segmentation. Topology metrics are calculated for arteries and veins at
the final test evaluation rather than after every training epoch.

These implementations are adaptations for this project's five-class label system. They are
not presented as the original authors' reference implementations.
