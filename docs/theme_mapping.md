# Photo-Culling Theme Mapping Sheet

## Purpose
This sheet converts messy annual discovery labels into a smaller, reusable master gallery taxonomy.

Use it as the bridge between:
- annual diary exploration,
- consolidated master gallery building,
- later script logic.

The goal is not perfect theory. The goal is consistent decisions.

---

## Working Source Path
The source for consolidation is the generated annual `theme_output` folders, not the original photo tree.

Current working arrangement:
- each annual `theme_output` folder sits where the script sits,
- these annual outputs can be copied to `/Volumes/All Photos/`,
- this places them at the same level as `dedupe_output`.

Recommended consolidation root:

`/Volumes/All Photos/`

Expected layout example:
- `/Volumes/All Photos/theme_output/2007`
- `/Volumes/All Photos/theme_output_/2010`
- `/Volumes/All Photos/theme_output/2015`
- `/Volumes/All Photos/dedupe_output`

If you prefer a tidier structure later, a dedicated parent such as `/Volumes/All Photos/theme_outputs/` would be even cleaner. But for now, `/Volumes/All Photos/` is a workable consolidation root.

Important distinction:
- original photo discovery may still run from `/Volumes/All Photos/Photos`,
- master-gallery consolidation should work from copied annual `theme_output` folders.

## Primary Rule
Each image can have:
- **one primary master category**,
- **zero, one, or more secondary categories**.

The primary category reflects the main viewing experience.
The secondary categories improve discovery and reuse.

Example:
- a piper on a harbour wall
  - primary: **People and Human Presence**
  - secondary: **Waterside and Harbour**

---

## Approved Master Categories

1. **Landscape**
2. **Waterside and Harbour**
3. **Nature Detail**
4. **Wildlife**
5. **Farm Animals**
6. **People and Human Presence**
7. **Place and Travel**
8. **Rural Life and Working Country**
9. **Weather, Light, and Atmosphere**

These are the categories to consolidate toward.

---

## Fast Decision Rules

### 1. People First Rule
If the human subject is the reason the image matters, the primary category is usually **People and Human Presence**.

Examples:
- musician playing in street
- police officer at event
- judge in ceremonial setting
- child on beach
- walker in city rain

The setting can still become a secondary category.

### 2. Setting First Rule
If the environment matters more than the people, choose the environmental category as primary.

Examples:
- tiny walkers in wide coastal view → **Landscape**
- person used mainly for scale in woodland → **Landscape** or **Rural Life and Working Country**

### 3. Wildlife Is Not Farm Animals
Wild animals belong in **Wildlife**.
Domesticated or agricultural animals belong in **Farm Animals**.

Examples:
- bear → **Wildlife**
- deer → **Wildlife**
- sheep → **Farm Animals**
- cow → **Farm Animals**
- horse depends on context:
  - wild or feral emphasis → **Wildlife**
  - agricultural/domestic context → **Farm Animals**

If uncertain, prefer **Wildlife** or a broad animal label internally rather than wrongly classifying as farm animal.

### 4. Travel Snapshot Safety Valve
Use **Place and Travel** when an image records a location well enough to be useful, but does not strongly qualify as a landscape, people image, or architecture-led subject.

### 5. Atmosphere Can Be Secondary
If an image is mainly about weather, mist, dramatic light, storminess, or season, **Weather, Light, and Atmosphere** can be primary.
Otherwise it is often secondary.

---

## Annual Label → Master Category Mapping

This table is the starting translation layer.

| Annual / Discovery Label | Primary Master Category | Possible Secondary Categories | Notes |
|---|---|---|---|
| coastal landscape photograph | Landscape | Waterside and Harbour; Weather, Light, and Atmosphere | Use harbour secondary only if structures or boats matter |
| harbour or port scene with boats | Waterside and Harbour | Landscape; Place and Travel; Weather, Light, and Atmosphere | Working waterfront, boats, piers, quays |
| countryside landscape photograph | Landscape | Rural Life and Working Country; Weather, Light, and Atmosphere | Use rural secondary when agricultural signs matter |
| woodland or forest scene | Landscape | Nature Detail; Weather, Light, and Atmosphere | Nature Detail if close, texture-led, or intimate |
| flower or plant close-up photograph | Nature Detail | Weather, Light, and Atmosphere | Usually close, texture, form, or colour driven |
| bird or wildlife photograph | Wildlife | Nature Detail; Weather, Light, and Atmosphere | Wildlife primary unless clearly habitat-led scenic image |
| farm animal photograph | Farm Animals | Rural Life and Working Country; People and Human Presence | Keep separate from wildlife |
| travel snapshot of place | Place and Travel | Landscape; Waterside and Harbour; People and Human Presence | Useful catch-all for documentary place images |
| village or town street scene | Place and Travel | People and Human Presence; Weather, Light, and Atmosphere | If people dominate, switch primary to People |
| market scene | People and Human Presence | Place and Travel | Human activity is usually the point |
| street performer | People and Human Presence | Place and Travel | If waterfront setting matters, add Waterside secondary |
| musician performing outdoors | People and Human Presence | Place and Travel; Waterside and Harbour | Role-based human category |
| police officer in scene | People and Human Presence | Place and Travel | Public-role images belong here |
| judge or ceremonial official | People and Human Presence | Place and Travel | Primary is role, not location |
| people in city | People and Human Presence | Place and Travel; Weather, Light, and Atmosphere | City is context unless place dominates |
| people at the beach | People and Human Presence | Landscape; Weather, Light, and Atmosphere | If beach is the real subject, swap primary |
| people in countryside | People and Human Presence | Rural Life and Working Country; Landscape | Depends whether person or environment dominates |
| beach scene with figures | Landscape | People and Human Presence; Weather, Light, and Atmosphere | Use People primary only when human story leads |
| dramatic sunset scene | Weather, Light, and Atmosphere | Landscape; Waterside and Harbour | Atmosphere primary if mood is the main point |
| misty countryside | Weather, Light, and Atmosphere | Landscape; Rural Life and Working Country | Mood-led images go here |
| boat close-up in harbour | Waterside and Harbour | Nature Detail; Place and Travel | Detail of working waterside object |
| farm scene with tractor, barns, fields | Rural Life and Working Country | Landscape; Farm Animals | Working countryside rather than pure landscape |
| grazing sheep in field | Farm Animals | Rural Life and Working Country; Landscape | Animal subject remains primary |
| bear in woodland | Wildlife | Landscape | Never Farm Animals |
| deer in mist | Wildlife | Weather, Light, and Atmosphere; Landscape | Wildlife primary unless atmosphere overwhelms subject |

---

## People and Human Presence Subcategories
Use these as internal tags or secondary labels inside the people category.

### Role-based
- musician
- performer
- police
- judge
- worker
- vendor
- fisher
- tourist
- child
- couple
- crowd

### Setting-based
- city
- beach
- countryside
- harbour
- festival
- street
- market

### Photograph type
- candid
- portrait in context
- event scene
- public life

These should usually remain sublabels, not top-level gallery families.

---

## Wildlife vs Farm Animals Rules

### Wildlife
Use for:
- birds
- deer
- foxes
- bears
- seals
- squirrels
- raptors
- wild ponies if treated as wild-living animals

### Farm Animals
Use for:
- sheep
- cows
- goats
- pigs
- chickens
- horses in clearly domestic or agricultural settings

### Uncertain Cases
If the classifier is uncertain:
- do not force a specific animal type,
- prefer broad animal grouping internally,
- avoid false precision.

Wrong-specific is worse than broad-correct.

---

## Primary/Secondary Examples

### Example 1
A violinist playing near moored fishing boats.
- primary: **People and Human Presence**
- secondary: **Waterside and Harbour**

### Example 2
A wide beach at dusk with two tiny walkers.
- primary: **Landscape**
- secondary: **People and Human Presence**
- secondary: **Weather, Light, and Atmosphere**

### Example 3
A cow standing in heavy morning mist.
- primary: **Farm Animals**
- secondary: **Weather, Light, and Atmosphere**
- secondary: **Rural Life and Working Country**

### Example 4
A deer crossing a road in forest light.
- primary: **Wildlife**
- secondary: **Landscape**

### Example 5
A harbour street with pedestrians, cafés, and strong sense of place.
- primary: **Place and Travel** or **People and Human Presence**
- choose **Place and Travel** if the location is the point,
- choose **People and Human Presence** if the human activity is the point.

---

## Suggested Script Logic
For future script development, the consolidation layer should do this:

1. keep the raw annual discovery label,
2. assign a primary master category,
3. optionally assign one or more secondary categories,
4. flag uncertain animal classifications,
5. allow later manual correction,
6. avoid collapsing everything into one rigid bucket too early.

Desirable output fields:
- file path
- year
- annual theme
- primary master category
- secondary categories
- confidence
- notes
- possible misclassification flag

---

## Path Guidance for Commands
For master-gallery consolidation, the working root should now be treated as:

`/Volumes/All Photos/`

That is where the copied annual `theme_output` folders can live, alongside `dedupe_output`.

Examples:
- `/Volumes/All Photos/theme_output/2007`
- `/Volumes/All Photos/theme_output/2008`
- `/Volumes/All Photos/theme_output/2010`

So there are now two separate path concepts:

### Discovery input root
Used when generating annual theme outputs from the original archive:

`/Volumes/All Photos/Photos`

### Consolidation input root
Used when building a cross-year master gallery from annual results:

`/Volumes/All Photos/`

A future consolidation script should point at the second of these, not the first.

## Immediate Next Build Goal
The next useful implementation step is a consolidation pass that:
- reads annual outputs,
- maps raw themes onto this approved taxonomy,
- supports primary and secondary categories,
- prevents wildlife/farm-animal mix-ups,
- produces one consolidated gallery report.

That is the bridge from annual diary browsing to a real master gallery.
