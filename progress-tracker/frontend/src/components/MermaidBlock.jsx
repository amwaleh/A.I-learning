import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#6366f1',
    primaryTextColor: '#e2e8f0',
    primaryBorderColor: '#4f46e5',
    lineColor: '#94a3b8',
    secondaryColor: '#1e293b',
    tertiaryColor: '#0f172a',
    background: '#0f172a',
    mainBkg: '#1e293b',
    nodeBorder: '#4f46e5',
    clusterBkg: '#1e293b',
    titleColor: '#e2e8f0',
    edgeLabelBackground: '#1e293b',
  },
  fontFamily: 'ui-sans-serif, system-ui, sans-serif',
  securityLevel: 'loose',
});

let idCounter = 0;

export default function MermaidBlock({ chart }) {
  const containerRef = useRef(null);
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!chart) return;

    const id = `mermaid-${idCounter++}`;
    mermaid.render(id, chart.trim())
      .then(({ svg }) => {
        setSvg(svg);
        setError(null);
      })
      .catch((err) => {
        console.error('Mermaid render error:', err);
        setError(chart);
        setSvg('');
      });
  }, [chart]);

  if (error) {
    return (
      <pre className="bg-slate-900 border border-red-800/50 rounded-lg p-4 overflow-x-auto text-sm text-slate-300">
        <code>{error}</code>
      </pre>
    );
  }

  return (
    <div
      ref={containerRef}
      className="my-4 flex justify-center overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
