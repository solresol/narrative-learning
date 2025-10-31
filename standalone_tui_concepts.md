# Standalone Narrative Learning UI Concepts

## 1. TUI Sketch
- **Top Bar**: Title "Narrative Learning Standalone" with status indicator (Connected / Stub Mode) and clock.
- **Left Column** (30% width):
  - *Overseer Panel*
    - Baseline explainable models listed with accuracy and Kendall Tau scores.
    - Colour legend showing Correct (green), Incorrect (red), Mixed Region (purple), and Unevaluated (amber).
    - Scrollable grid of data points; glyph colour reflects combined correctness from overseer history.
  - *Hypothesis History*
    - Chronological list with timestamp, short description, and key metric deltas.
    - Selecting a hypothesis recolours the overseer grid and populates the detail pane.
- **Centre Column** (40% width):
  - *Active Hypothesis Detail* (labelled "Overseer")
    - Prompt snippet, rationale, and selected metrics.
    - Tabs for "Errors", "Supporting Examples", and "Notes".
  - *Underling Progress* (labelled "Underling")
    - Horizontal progress bar reflecting evaluation completion.
    - Scrollable coordinate list with chevron showing the current data point under review.
    - Async status ticker (e.g., "Evaluating point 12/50").
- **Right Column** (30% width):
  - *Dataset Overview*
    - Histogram or sparkline for feature distributions.
    - Summary stats (rows, classes, current round).
  - *Event Log*
    - Colour-coded entries for info/warn/error.
    - Filters toggled via hotkeys.
- **Footer**: Shortcut reminders and message area for background task notifications.

## 2. HTML/CSS UI Concept
- **Layout**: Responsive three-column grid using CSS Grid, collapsing to stacked panels on mobile.
- **Header**: Fixed bar with application name, stub/online toggle, and asynchronous task indicator.
- **Left Sidebar (Overseer)**:
  - Card per baseline model showing accuracy, Kendall Tau, and mini sparkline.
  - Interactive scatter plot overlaying correctness colours and mixed-region shading.
  - Hypothesis history list with badges indicating evaluation status (evaluated, pending, unevaluated).
- **Central Workspace**:
  - Tabs for "Current Hypothesis", "Error Analysis", and "Notes".
  - Prominent progress bar (Underling) with animated stripe and status text.
  - Scrollable table of coordinates; the active row is highlighted with a pulsing marker.
  - Async toasts appear in the corner to confirm completed evaluations.
- **Right Sidebar**:
  - Dataset summary cards with metrics and mini charts.
  - Event log with filter pills and search box.
- **Footer**: Command palette hint ("Press / to search"), plus quick links to documentation and export actions.

## 3. Demo Narrative
1. **Introduction**
   - Explain the goal: showcase narrative learning without PostgreSQL, focusing on transparent hypothesis tracking.
   - Highlight the two key roles: Overseer (historical context + baseline models) and Underling (active evaluation).
2. **Dataset Setup**
   - Import a small CSV and emphasise the two-feature constraint.
   - Show the overseer scatter plot initial state (all unevaluated amber).
3. **Baseline Model Overview**
   - Trigger baseline explainable models; point out displayed accuracy and Kendall Tau scores.
   - Discuss colour coding of data points, noting mixed regions where models disagree.
4. **Hypothesis Iteration**
   - Generate a new hypothesis; watch the Underling progress bar advance asynchronously.
   - Narrate the coordinate list scrolling as each point is evaluated.
   - Pause interaction to demonstrate responsiveness while computations run.
5. **History Review**
   - Select previous hypotheses; observe recoloured data points and stored rationale.
   - Emphasise Overseer history as a teaching tool for model evolution.
6. **Wrap-Up**
   - Summarise metrics, export results, and point to extensibility hooks.

## 4. Alternative Demonstrations of Narrative Learning
- **Notebook Walkthrough**: Use Jupyter to step through dataset import, hypothesis generation, and evaluation with visualisations.
- **Static Report**: Generate a narrative PDF or HTML report combining metrics, hypothesis history, and annotated data visualisations.
- **Interactive Web Demo**: Host a lightweight web app with precomputed scenarios allowing users to toggle hypotheses and view outcomes.
- **Live Coding Session**: Implement a simple hypothesis evaluator from scratch, narrating the algorithmic steps.
- **Data Storytelling Workshop**: Provide printed hypothesis cards and coloured tokens representing correctness; participants act as Overseer/Underling to simulate the workflow.
