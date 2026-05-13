import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Tracks scroll progress and time spent on a scrollable container.
 * Returns { scrollPercent, timeSpent, containerRef, reset }
 */
export default function useScrollProgress() {
  const containerRef = useRef(null);
  const [scrollPercent, setScrollPercent] = useState(0);
  const [timeSpent, setTimeSpent] = useState(0);
  const startTimeRef = useRef(Date.now());
  const timerRef = useRef(null);
  const maxScrollRef = useRef(0);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;

    const { scrollTop, scrollHeight, clientHeight } = el;
    const maxScroll = scrollHeight - clientHeight;
    if (maxScroll <= 0) {
      maxScrollRef.current = 100;
      setScrollPercent(100);
      return;
    }
    const percent = Math.min(100, Math.round((scrollTop / maxScroll) * 100));
    // High-water mark — never decrease
    if (percent > maxScrollRef.current) {
      maxScrollRef.current = percent;
      setScrollPercent(percent);
    }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    el.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => el.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  // Time tracker — updates every 5 seconds
  useEffect(() => {
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setTimeSpent(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 5000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const reset = useCallback(() => {
    maxScrollRef.current = 0;
    setScrollPercent(0);
    setTimeSpent(0);
    startTimeRef.current = Date.now();
  }, []);

  return { scrollPercent, timeSpent, containerRef, reset };
}
