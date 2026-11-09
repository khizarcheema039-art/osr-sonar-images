import React from 'react';

export default function Figure({ figure, onOpenLightbox, className = '', breakout = false }) {
  if (!figure) return null;

  return (
    <figure 
      className={`my-8 ${breakout ? 'w-full lg:-mx-6 lg:w-[calc(100%+3rem)]' : 'w-full'} ${className}`}
    >
      <div 
        className="group relative cursor-pointer border border-rule dark:border-rule-dark bg-panel overflow-hidden"
        onClick={() => onOpenLightbox && onOpenLightbox(figure)}
        title="Click to view full image in lightbox"
      >
        <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity bg-panel/90 text-accent font-mono text-[10px] px-2 py-0.5 border border-rule/30 pointer-events-none">
          Click to expand ↗
        </div>

        <img
          src={figure.file}
          alt={figure.caption || figure.title}
          loading="lazy"
          className="w-full h-auto object-contain block mx-auto transition-transform duration-200 group-hover:scale-[1.008]"
        />
      </div>

      <figcaption className="mt-2.5 text-xs text-muted dark:text-muted-dark leading-relaxed font-sans">
        <span className="font-mono font-semibold text-accent dark:text-accent-dark mr-1.5">
          {figure.number}:
        </span>
        <span className="text-ink/90 dark:text-ink-dark/90 font-medium mr-1.5">
          {figure.title}.
        </span>
        <span className="text-muted dark:text-muted-dark">
          {figure.caption}
        </span>
      </figcaption>
    </figure>
  );
}
