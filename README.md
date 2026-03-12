# dlc-loe-detector
Automated loss-of-equilibrium (LOE) detection system for fish thermal tolerance assays, combining DeepLabCut-based pose estimation with a ResNet34 classification model.

## Overview
This repository contains the code used in the above study. The system automatically detects the timing of loss of equilibrium (LOE) in fish during thermal stress tests, replacing subjective manual observation with an objective, high-throughput pipeline.
The pipeline consists of two main components:

1. Keypoint detection — DeepLabCut-based pose estimation with region partitioning and color transformation preprocessing
2. LOE classification — ResNet34-based frame classification combined with keypoint coordinates, followed by time-series post-processing
