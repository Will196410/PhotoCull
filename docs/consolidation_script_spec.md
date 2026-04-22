# Master Gallery Consolidation Script Spec

## Purpose
This document defines the next major build step for the photo-culling project: a script that reads annual theme outputs and produces a consolidated cross-year master gallery view.

This is the bridge between:
- annual diary discovery,
- category consolidation,
- a reusable master gallery.

The annual theme script already discovers useful clusters within a year.
What is missing is the layer that combines those results across years.

---

## Goal
Build a script that:

1. reads annual outputs from `theme_output/`,
2. loads `*_themes.csv` and `*_images.csv` for each year,
3. maps rough annual discovery labels into approved master categories,
4. supports one primary category and optional secondary categories,
5. flags likely misclassifications,
6. produces consolidated CSV outputs,
7. optionally builds a reviewable HTML gallery grouped by master category.

---

## Canonical Inputs

### Project root
The project should assume a configurable project root.

Example on the current system:

```text
/Volumes/All Photos/
