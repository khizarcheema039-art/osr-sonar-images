import React, { useState } from 'react';
import { OSR_METHODS } from '../data/results';

const UNKNOWN_CLASSES = [
  { key: 'chain', label: 'Chain' },
  { key: 'valve', label: 'Valve' },
  { key: 'shampoo-bottle', label: 'Shampoo Bottle' },
  { key: 'standing-bottle', label: 'Standing Bottle' }
];

export default function PerClassChart({ selectedMethod, onSelectMethod, hoveredMethod, onHoverMethod }) {
  const [tooltip, setTooltip] = useState(null);

  const methodsList = [
    { key: 'MSP', label: 'MSP', color: '#6B7075' },
    { key: 'SVDD', label: 'Deep SVDD', color: '#C9822B' },
    { key: 'MetricLearning', label: 'Metric Learning', color: '#3F7F78' }
  ];

  // Dimensions
  const height = 260;
  const paddingLeft = 45;
  const paddingRight = 20;
  const paddingTop = 25;
  const paddingBottom = 40;
  const chartWidth = 600; // viewBox width

  const plotWidth = chartWidth - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;

  const groupWidth = plotWidth / UNKNOWN_CLASSES.length;
  const barWidth = 14;
  const barSpacing = 4;
  const totalBarsWidth = methodsList.length * barWidth + (methodsList.length - 1) * barSpacing;

  return (
    <div className="border border-rule dark:border-rule-dark bg-paper dark:bg-panel p-4 my-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-2 border-b border-rule/50 dark:border-rule-dark/50">
        <div>
          <h4 className="font-mono text-xs font-semibold text-ink dark:text-ink-dark uppercase tracking-wider">
            Per-Unknown Class Unknown Detection Rate (%)
          </h4>
          <span className="text-[11px] text-muted dark:text-muted-dark font-sans">
            Proportion of novel class acoustic test instances successfully rejected
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          {methodsList.map((m) => {
            const isSelected = selectedMethod === m.key;
            return (
              <button
                key={m.key}
                onClick={() => onSelectMethod(m.key)}
                onMouseEnter={() => onHoverMethod && onHoverMethod(m.key)}
                onMouseLeave={() => onHoverMethod && onHoverMethod(null)}
                className={`flex items-center gap-1.5 transition-opacity ${
                  selectedMethod === m.key ? 'opacity-100 font-semibold' : 'opacity-70 hover:opacity-100'
                }`}
              >
                <span className="w-2.5 h-2.5 inline-block" style={{ backgroundColor: m.color }}></span>
                <span className="text-ink dark:text-ink-dark">{m.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* SVG Bar Chart */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${chartWidth} ${height}`}
          className="w-full h-auto max-w-full text-xs font-mono select-none"
          role="img"
          aria-label="Grouped bar chart showing per-unknown class detection rates"
        >
          {/* Y Axis Grid lines (0, 25, 50, 75, 100) */}
          {[0, 25, 50, 75, 100].map((val) => {
            const y = paddingTop + plotHeight - (val / 100) * plotHeight;
            return (
              <g key={val}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={chartWidth - paddingRight}
                  y2={y}
                  stroke="#D9D4C8"
                  strokeDasharray={val === 0 ? "none" : "2,3"}
                  strokeWidth="1"
                  className="dark:stroke-[#2A3035]"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  className="fill-muted dark:fill-muted-dark text-[10px]"
                >
                  {val}%
                </text>
              </g>
            );
          })}

          {/* Groups */}
          {UNKNOWN_CLASSES.map((cls, groupIdx) => {
            const groupCenterX = paddingLeft + groupIdx * groupWidth + groupWidth / 2;
            const startX = groupCenterX - totalBarsWidth / 2;

            return (
              <g key={cls.key}>
                {/* Bars */}
                {methodsList.map((m, mIdx) => {
                  const val = OSR_METHODS[m.key].per_class_detection[cls.key];
                  const barH = (val / 100) * plotHeight;
                  const barX = startX + mIdx * (barWidth + barSpacing);
                  const barY = paddingTop + plotHeight - barH;
                  const isHighlighted = (hoveredMethod || selectedMethod) === m.key;

                  return (
                    <g 
                      key={m.key}
                      className="cursor-pointer transition-all duration-150"
                      onClick={() => onSelectMethod(m.key)}
                      onMouseEnter={() => {
                        onHoverMethod && onHoverMethod(m.key);
                        setTooltip({
                          x: barX + barWidth / 2,
                          y: barY - 8,
                          text: `${m.label}: ${val}% (${cls.label})`
                        });
                      }}
                      onMouseLeave={() => {
                        onHoverMethod && onHoverMethod(null);
                        setTooltip(null);
                      }}
                    >
                      <rect
                        x={barX}
                        y={barY}
                        width={barWidth}
                        height={barH}
                        fill={m.color}
                        opacity={isHighlighted ? 1 : 0.45}
                        stroke={isHighlighted ? '#15191C' : 'none'}
                        strokeWidth="1"
                        rx="1"
                      />
                      {/* Numeric value atop bar */}
                      <text
                        x={barX + barWidth / 2}
                        y={barY - 4}
                        textAnchor="middle"
                        className={`text-[9px] font-mono ${
                          isHighlighted ? 'fill-accent font-semibold' : 'fill-muted/70 opacity-60'
                        }`}
                      >
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* X Axis Class Label */}
                <text
                  x={groupCenterX}
                  y={paddingTop + plotHeight + 18}
                  textAnchor="middle"
                  className="fill-ink dark:fill-ink-dark text-[11px] font-sans font-medium"
                >
                  {cls.label}
                </text>
              </g>
            );
          })}

          {/* Tooltip Overlay */}
          {tooltip && (
            <g transform={`translate(${tooltip.x}, ${tooltip.y})`} pointerEvents="none">
              <rect
                x="-60"
                y="-22"
                width="120"
                height="20"
                fill="#15191C"
                rx="2"
                className="stroke-rule/50"
              />
              <text
                x="0"
                y="-9"
                textAnchor="middle"
                fill="#F4F1EA"
                className="text-[10px] font-mono font-medium"
              >
                {tooltip.text}
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
