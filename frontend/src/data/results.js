/**
 * Benchmark results and configuration data for
 * "Open-Set Recognition Using Underwater SONAR Images"
 *
 * All numeric values correspond strictly to outputs/results/all_results.json
 * and benchmark_summary.csv.
 */

export const PROJECT_METADATA = {
  title: "Open-Set Recognition Using Underwater SONAR Images",
  degree: "Final Year Project",
  institution: "Department of Computer Science, Bahria University, Islamabad",
  authors: [
    { name: "Mohammad Khizar Hayat Cheema", id: "01-134231-039" },
    { name: "Muhammad Haroon", id: "01-134231-052" }
  ],
  supervisor: "Dr. Arif Ur Rahman",
  dataset: "Marine Debris Forward-Looking Sonar (MD-FLS, ARIS Explorer 3000)",
  backbone: "ResNet-50 (2048-D feature embedding)",
  preprocessing: {
    median_filter: "3×3 kernel",
    clahe: "clip limit 2.0, tile grid 8×8",
    input_size: "224×224",
    normalization: "ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]"
  }
};

export const CLASSES = {
  known: [
    "bottle",
    "can",
    "tire",
    "drink-carton",
    "hook",
    "propeller"
  ],
  unknown: [
    "chain",
    "valve",
    "shampoo-bottle",
    "standing-bottle"
  ]
};

export const BASELINE_METRICS = {
  model: "ResNet-50",
  validation_accuracy: 0.9962,
  test_known_accuracy: 0.9924,
  f1_score: 0.992,
  epochs: 25,
  lr: "1e-4",
  batch_size: 32
};

export const OSR_METHODS = {
  MSP: {
    key: "MSP",
    index: "1",
    label: "Maximum Softmax Probability (MSP)",
    shortLabel: "MSP",
    auroc: 0.8165,
    ap: 0.5782,
    tpr: 0.8487,
    fpr: 0.3177,
    udr: 0.6823,
    threshold: 0.9887,
    thresholdRange: [0.5, 1.0],
    thresholdStep: 0.001,
    per_class_detection: {
      chain: 60,
      valve: 85,
      "shampoo-bottle": 57,
      "standing-bottle": 63
    },
    formula: "g(x) = max_c P(y=c | x) >= tau",
    decisionExplanation: "Accepts as known if the maximum softmax posterior probability exceeds tau = 0.9887; rejects as novel/unknown otherwise.",
    limitation: "Standard cross-entropy produces overconfident posteriors even for completely novel acoustic signatures due to unbounded softmax exponentiation.",
    dashboardFig: "msp_complete_dashboard.png",
    dashboardCaption: "Fig. 6A: MSP performance curves, distribution separation, and rejection rates across unknown classes."
  },
  SVDD: {
    key: "SVDD",
    index: "2",
    label: "Deep SVDD (nearest centroid)",
    shortLabel: "Deep SVDD",
    auroc: 0.8986,
    ap: 0.8275,
    tpr: 0.8413,
    fpr: 0.1171,
    udr: 0.8829,
    threshold: -0.684,
    thresholdRange: [-1.8, 0.0],
    thresholdStep: 0.005,
    per_class_detection: {
      chain: 76,
      valve: 93,
      "shampoo-bottle": 99,
      "standing-bottle": 100
    },
    formula: "g(x) = - min_c ||phi(x) - mu_c||_2 >= tau",
    decisionExplanation: "Computes the L2 Euclidean distance between the ResNet-50 2048-D representation phi(x) and class centroids mu_c. Accepts if familiarity score -d_min >= -0.684.",
    limitation: "Features are optimized via baseline cross-entropy without an explicit metric compactness penalty during training; representations may disperse non-uniformly across classes.",
    dashboardFig: "svdd_complete_dashboard.png",
    dashboardCaption: "Fig. 6B: Deep SVDD centroid distance distributions, ROC/PR curves, and per-unknown class detection rates."
  },
  MetricLearning: {
    key: "MetricLearning",
    index: "3",
    label: "Deep Metric Learning (triplet loss)",
    shortLabel: "Metric Learning",
    auroc: 0.9255,
    ap: 0.8448,
    tpr: 0.9410,
    fpr: 0.2525,
    udr: 0.7475,
    threshold: -0.712,
    thresholdRange: [-1.8, 0.0],
    thresholdStep: 0.005,
    per_class_detection: {
      chain: 71,
      valve: 76,
      "shampoo-bottle": 78,
      "standing-bottle": 74
    },
    formula: "g(x) = - min_c ||f(x) - c_k||_2 >= tau",
    decisionExplanation: "Maps images into a 128-D L2-normalized metric hypersphere optimized with Triplet Margin Loss and Multi-Similarity mining. Accepts if score -d_min >= -0.712.",
    limitation: "While achieving the highest AUROC and TPR, the metric margin occasionally projects high-frequency specular sonar noise near peripheral known clusters, resulting in a moderate FPR of 25.25%.",
    dashboardFig: "dml_complete_dashboard.png",
    dashboardCaption: "Fig. 6C: Deep Metric Learning cluster compactness, margin separation, and open-set rejection dynamics."
  }
};

export const REFERENCES = [
  {
    cite: "Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., & Boult, T. E. (2013)",
    title: "Toward Open Set Recognition",
    venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)",
    note: "Formalised the open-space risk formulation and foundational open-set recognition bounds."
  },
  {
    cite: "Hendrycks, D., & Gimpel, K. (2017)",
    title: "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks",
    venue: "International Conference on Learning Representations (ICLR)",
    note: "Introduced Maximum Softmax Probability (MSP) as the universal baseline for out-of-distribution detection."
  },
  {
    cite: "Bendale, A., & Boult, T. E. (2016)",
    title: "Towards Open Set Deep Networks",
    venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    note: "Introduced OpenMax and Extreme Value Theory (EVT) Weibull calibration for deep neural networks."
  },
  {
    cite: "Perera, P., & Patel, V. M. (2021)",
    title: "One-Class Classification: A Survey",
    venue: "IEEE Transactions on Neural Networks and Learning Systems (TNNLS)",
    note: "Comprehensive survey of SVDD, centroid modeling, and one-class boundaries."
  },
  {
    cite: "Valdenegro-Toro, M. (2019)",
    title: "Subsea Object Classification in Forward-Looking Sonar Images Using Deep Convolutional Neural Networks",
    venue: "IEEE Journal of Oceanic Engineering (JOE)",
    note: "Established acoustic noise challenges, speckle, and CNN baselines for forward-looking sonar imagery."
  },
  {
    cite: "He, K., Zhang, X., Ren, S., & Sun, J. (2016)",
    title: "Deep Residual Learning for Image Recognition",
    venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    note: "The ResNet-50 architecture used as the feature backbone throughout our study."
  }
];

export const FIGURES_CATALOG = [
  {
    id: "sample_images",
    file: "/figures/sample_images.png",
    number: "Fig. 1",
    title: "Known Classes in Marine Debris FLS Dataset",
    caption: "Acoustic samples of the six known classes (bottle, can, tire, drink-carton, hook, propeller) captured with the ARIS Explorer 3000 FLS."
  },
  {
    id: "unknown_sample_images",
    file: "/figures/unknown_sample_images.png",
    number: "Fig. 2",
    title: "Unknown (Novel) Classes Omitted During Training",
    caption: "Acoustic signatures of the four novel test classes (chain, valve, shampoo-bottle, standing-bottle) withheld during model optimization."
  },
  {
    id: "class_distribution",
    file: "/figures/class_distribution.png",
    number: "Fig. 3",
    title: "Class Distribution & Partition Split",
    caption: "Sample frequency distribution across known and novel target categories within the watertight and turntable test environments."
  },
  {
    id: "preprocessing_example",
    file: "/figures/preprocessing_example.png",
    number: "Fig. 4",
    title: "Acoustic Preprocessing Pipeline",
    caption: "Stage-by-stage acoustic enhancement: raw acoustic FLS image -> 3x3 median filter (speckle attenuation) -> CLAHE (clip 2.0, tile 8x8) -> ImageNet standardisation."
  },
  {
    id: "confusion_matrix_baseline",
    file: "/figures/confusion_matrix_baseline.png",
    number: "Fig. 5A",
    title: "Closed-Set ResNet-50 Baseline Confusion Matrix",
    caption: "Normalized test confusion matrix on the 6 known classes demonstrating 99.24% test accuracy under closed-world assumptions."
  },
  {
    id: "resnet50_baseline_curves",
    file: "/figures/resnet50_baseline_curves.png",
    number: "Fig. 5B",
    title: "Closed-Set ResNet-50 Training & Validation Curves",
    caption: "Loss and accuracy trajectories over 25 epochs showing rapid convergence and high generalization on known classes."
  },
  {
    id: "all_methods_comparison",
    file: "/figures/all_methods_comparison.png",
    number: "Fig. 6",
    title: "ROC & Precision-Recall Curves Across All Methods",
    caption: "Comparative ROC curves: Deep Metric Learning achieves AUROC 0.9255, Deep SVDD achieves 0.8986, and MSP reaches 0.8165."
  },
  {
    id: "all_methods_unknown_detection",
    file: "/figures/all_methods_unknown_detection.png",
    number: "Fig. 7",
    title: "Per-Unknown-Class Rejection Rate Comparison",
    caption: "Empirical rejection rates per unknown category under each method's operating threshold. Deep SVDD leads on bottles, while Metric Learning provides balanced rejection."
  },
  {
    id: "final_benchmark_dashboard",
    file: "/figures/final_benchmark_dashboard.png",
    number: "Fig. 8",
    title: "Complete Open-Set Recognition Master Dashboard",
    caption: "Consolidated benchmark synthesis: AUROC, AP, TPR/FPR trade-offs, per-class rejections, and score distributions."
  },
  {
    id: "dml_tsne_visualization",
    file: "/figures/dml_tsne_visualization.png",
    number: "Fig. 9A",
    title: "t-SNE Representation in Metric Space (128-D Projection)",
    caption: "Two-dimensional t-SNE projection of the 128-D metric embedding space. Note the compact, isolated clusters for known classes and the spatial divergence of unknown acoustic signatures."
  },
  {
    id: "metric_tsne",
    file: "/figures/metric_tsne.png",
    number: "Fig. 9B",
    title: "t-SNE Metric Space Distribution with Unknowns",
    caption: "Detailed inspection of the metric geometry illustrating how multi-similarity triplet mining forces known classes into hyper-spherical boundaries."
  }
];
