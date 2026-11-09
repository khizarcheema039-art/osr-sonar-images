import React from 'react';
import Figure from './Figure';
import { CLASSES, PROJECT_METADATA } from '../data/results';

export default function DataSplit({ onOpenLightbox }) {
  return (
    <div className="space-y-6">
      <p className="text-sm text-ink/90 dark:text-ink-dark/90 leading-relaxed font-sans">
        We partition the Marine Debris Forward-Looking Sonar (MD-FLS) dataset into strictly disjoint known and unknown sets. 
        The known set comprises six object classes present in the training split, whereas the unknown set consists of four distinct 
        classes strictly omitted during all training phases and presented exclusively during open-set inference to evaluate rejection capability.
      </p>

      {/* Class partitions as two plain lists */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 my-6">
        {/* Known Classes */}
        <div className="border border-rule dark:border-rule-dark p-4 bg-paper dark:bg-panel">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-rule dark:border-rule-dark">
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-teal-sonar">
              Known Classes (Train & Test)
            </span>
            <span className="font-mono text-xs text-muted dark:text-muted-dark">K = 6</span>
          </div>
          <ul className="space-y-1.5 font-mono text-xs">
            {CLASSES.known.map((cls, idx) => (
              <li key={cls} className="flex items-center justify-between py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-teal-sonar inline-block"></span>
                  <span className="text-ink dark:text-ink-dark font-medium">{cls}</span>
                </span>
                <span className="text-muted dark:text-muted-dark text-[11px]">class_idx: {idx}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] text-muted dark:text-muted-dark font-sans leading-normal">
            Used to train the ResNet-50 feature extractor and baseline closed-set classifier.
          </p>
        </div>

        {/* Unknown Classes */}
        <div className="border border-rule dark:border-rule-dark p-4 bg-paper dark:bg-panel">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-rule dark:border-rule-dark">
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-accent">
              Unknown / Novel Classes (Test Only)
            </span>
            <span className="font-mono text-xs text-muted dark:text-muted-dark">U = 4</span>
          </div>
          <ul className="space-y-1.5 font-mono text-xs">
            {CLASSES.unknown.map((cls) => (
              <li key={cls} className="flex items-center justify-between py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent inline-block"></span>
                  <span className="text-ink dark:text-ink-dark font-medium">{cls}</span>
                </span>
                <span className="text-accent text-[11px]">label: -1 (novel)</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] text-muted dark:text-muted-dark font-sans leading-normal">
            Withheld completely during optimization; utilized as positive targets for unknown rejection.
          </p>
        </div>
      </div>

      {/* Figures: Sample Images & Class Distribution */}
      <Figure
        figure={{
          file: "/figures/sample_images.png",
          number: "Fig. 1",
          title: "Known Class Acoustic Exemplars",
          caption: "Representative forward-looking sonar crops for bottle, can, tire, drink-carton, hook, and propeller."
        }}
        onOpenLightbox={onOpenLightbox}
      />

      <Figure
        figure={{
          file: "/figures/unknown_sample_images.png",
          number: "Fig. 2",
          title: "Novel Acoustic Targets (Unknowns)",
          caption: "Test instances for chain, valve, shampoo-bottle, and standing-bottle. Note the severe speckle noise and variable acoustic shadows."
        }}
        onOpenLightbox={onOpenLightbox}
      />

      <Figure
        figure={{
          file: "/figures/class_distribution.png",
          number: "Fig. 3",
          title: "Sample Frequency Distribution",
          caption: "Distribution of bounding-box crops across known and novel categories extracted from ARIS Explorer 3000 sequences."
        }}
        onOpenLightbox={onOpenLightbox}
      />

      {/* Preprocessing Pipeline */}
      <div className="my-8 border-t border-rule dark:border-rule-dark pt-6">
        <h4 className="font-serif text-base font-semibold text-ink dark:text-ink-dark mb-2">
          Acoustic Preprocessing Pipeline
        </h4>
        <p className="text-xs text-muted dark:text-muted-dark font-sans mb-4 leading-relaxed">
          Forward-looking sonar frames suffer from acoustic speckle, non-uniform beam attenuation, and high reflectivity clutter. 
          Each cropped region of interest is conditioned using the following strict deterministic sequence:
        </p>

        {/* Monospace Parameters Table */}
        <div className="border border-rule dark:border-rule-dark overflow-hidden mb-6 font-mono text-xs">
          <table className="w-full text-left">
            <thead className="bg-paper-dark text-paper border-b border-rule/30">
              <tr>
                <th className="p-2.5 font-semibold">Stage</th>
                <th className="p-2.5 font-semibold">Operation</th>
                <th className="p-2.5 font-semibold">Exact Parameters</th>
                <th className="p-2.5 font-semibold">Purpose</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-rule/40 dark:divide-rule-dark/40 bg-paper dark:bg-panel text-ink dark:text-ink-dark">
              <tr>
                <td className="p-2.5 text-accent font-semibold">01</td>
                <td className="p-2.5 font-medium">Median Filter</td>
                <td className="p-2.5 text-muted dark:text-muted-dark">ksize = 3×3</td>
                <td className="p-2.5 text-[11px] font-sans">Suppresses high-frequency multiplicative acoustic speckle</td>
              </tr>
              <tr>
                <td className="p-2.5 text-accent font-semibold">02</td>
                <td className="p-2.5 font-medium">CLAHE</td>
                <td className="p-2.5 text-muted dark:text-muted-dark">clip_limit = 2.0, tile_grid = (8, 8)</td>
                <td className="p-2.5 text-[11px] font-sans">Equalizes local acoustic shadow and specular contrast</td>
              </tr>
              <tr>
                <td className="p-2.5 text-accent font-semibold">03</td>
                <td className="p-2.5 font-medium">Spatial Resize</td>
                <td className="p-2.5 text-muted dark:text-muted-dark">224 × 224 pixels (bilinear)</td>
                <td className="p-2.5 text-[11px] font-sans">Standardizes dimensions for deep CNN backbone receptive field</td>
              </tr>
              <tr>
                <td className="p-2.5 text-accent font-semibold">04</td>
                <td className="p-2.5 font-medium">ImageNet Norm</td>
                <td className="p-2.5 text-muted dark:text-muted-dark">mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]</td>
                <td className="p-2.5 text-[11px] font-sans">Conditions tensor distribution for pretrained transfer weights</td>
              </tr>
            </tbody>
          </table>
        </div>

        <Figure
          figure={{
            file: "/figures/preprocessing_example.png",
            number: "Fig. 4",
            title: "Acoustic Preprocessing Before and After",
            caption: "Comparison showing raw sonar ROI input, median filtered output, and post-CLAHE acoustic contrast enhancement."
          }}
          onOpenLightbox={onOpenLightbox}
        />
      </div>
    </div>
  );
}
