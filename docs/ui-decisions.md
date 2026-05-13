# UI Design System & Layout Decisions

> **This is a living document.** Every UI/layout change MUST be recorded here with the date, what changed, and why. This ensures design continuity and helps future contributors understand the rationale behind the current UI.

---

## Current Layout Architecture

### App Shell (`Layout.jsx`)
- **Top navbar**: Sticky, backdrop-blur, `max-w-screen-2xl` container
- **Content area**: `max-w-screen-2xl` with responsive padding
- **Theme**: Dark slate (`bg-slate-950`) with indigo accents

### Dashboard (`Dashboard.jsx`)
- Grid of `ProjectCard` components
- Overall progress header with percentage and streak
- Cards show: title, description, topic count, progress bar

### Project Detail (`ProjectDetail.jsx`) — Split Panel
```
┌──────────────────────────────────────────────────────────┐
│  ← Back    Project Title              12/20 topics  60% │
├────────────────┬─────────────────────────────────────────┤
│ TOPICS (320px) │  Content Area (flex-1)                  │
│ Collapsible    │  Scrollable markdown                    │
│ sidebar        │  or empty state placeholder             │
└────────────────┴─────────────────────────────────────────┘
```

### Color System
| Token | Usage |
|-------|-------|
| `slate-950` | Page background |
| `slate-900/60` | Card/panel backgrounds |
| `slate-800` | Borders |
| `indigo-400/500` | Primary accent, links, active state |
| `emerald-400` | Completed status |
| `amber-400` | In-progress status |
| `slate-500/600` | Not-started status |
| `rose-400` | Destructive actions (logout) |

### Component Modes
| Component | Props | Behavior |
|-----------|-------|----------|
| `TopicItem` | `compact={true}` | Sidebar: small text, truncated, no dropdown |
| `TopicItem` | `compact={false}` | Full-width with dropdown selector |
| `TopicContent` | `fullHeight={true}` | Fills parent, sticky header, scrollable |
| `TopicContent` | `fullHeight={false}` | Standalone card with margin |

---

## Change Log

### 2026-05-13 — Split Panel Layout for Project Detail
**Changed:** Redesigned ProjectDetail from vertical stack to horizontal split panel layout.

**Before:** Topics listed in a card, content rendered below in a separate card — required scrolling past topics to see content.

**After:**
- Left sidebar (320px, collapsible) with compact topic tree
- Right content area (flex-1) fills remaining space with markdown content
- Both panels scroll independently at full viewport height (`calc(100vh - 10rem)`)
- Sidebar collapses to 48px icon strip via `PanelLeftClose`/`PanelLeftOpen` toggle
- Empty state with icon + instructional text when no topic selected
- Layout container widened from `max-w-7xl` to `max-w-screen-2xl`
- Header compacted to single row (back button, title, progress inline)

**Why:** Vertical stacking forced users to scroll past the topic list every time they selected a new topic. The split panel keeps navigation persistent alongside content — matching the documentation-reader pattern users expect from learning platforms.

**Files modified:**
- `frontend/src/pages/ProjectDetail.jsx` — full rewrite to split layout
- `frontend/src/components/TopicItem.jsx` — added `compact` mode
- `frontend/src/components/TopicContent.jsx` — added `fullHeight` mode
- `frontend/src/components/Layout.jsx` — widened to `max-w-screen-2xl`

---

*Add new entries above this line, newest first.*
