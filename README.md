# VisiGUI

### Gesture-driven graphical project exploration with an interactive generative 3D world.

VisiGUI is the graphical interface for [VisiCLI](https://github.com/AdolfMacro/VisiCLI).

It combines static Python project analysis, gesture-based interaction, project navigation, and an interactive generative 3D environment in a single graphical application.

<p align="center">
  <img src="https://raw.githubusercontent.com/AdolfMacro/VisiGUI/main/screenshots/SC01.png" alt="VisiGUI Screenshot" width="900">
</p>

---

## Overview

VisiGUI is built around two connected parts:

* **Project Explorer** — visual exploration of Python projects using the same core analysis logic as VisiCLI.
* **Generative 3D World** — an interactive visual environment controlled through hand gestures.

The project is designed so that gesture recognition does not directly execute project operations. Gestures are interpreted as semantic input and then converted into explicit interactions by the controller layer.

This keeps the interaction model separate from the actual project analysis and execution logic.

---

## Core Features

### Python Project Analysis

VisiGUI uses static analysis to inspect Python projects without importing or executing the target project.

The analysis layer builds a structured representation of the project, including:

* Projects
* Packages
* Modules
* Classes
* Functions
* Files
* Relationships between components

This structure is represented as a `ProjectGraph`.

---

### Gesture Interaction

The application can use hand gestures as an interaction layer.

The important distinction is:

```text
Gesture ≠ Action
```

A detected gesture represents user intent.
The controller decides what that intent means in the current context.

For example:

```text
Gesture.FIVE
      ↓
Gesture Event
      ↓
Controller
      ↓
Context-aware Action
```

This prevents the vision layer from becoming tightly coupled to project operations or the 3D simulation.

---

### Generative 3D World

VisiGUI also contains a separate generative 3D environment.

The environment can be explored through hand movement and gestures, including:

* Moving through the generated environment
* Zooming
* Zooming out
* Changing the explored area
* Interacting with generated visual elements

The 3D world is intentionally separated from the project graph so that both systems can evolve independently.

---

## Architecture

The high-level architecture is organized around shared interaction logic and separate visual environments.

```text
                    ┌─────────────────────┐
                    │      Camera         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Hand Detection    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Gesture Recognition │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Controller      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │  Project Graph   │   │ Generative World │
          │                  │   │                  │
          │ Files            │   │ 3D Environment   │
          │ Classes          │   │ Navigation       │
          │ Functions        │   │ Zoom             │
          │ Packages         │   │ Interaction      │
          └──────────────────┘   └──────────────────┘
```

The main design principle is to keep these responsibilities separate:

```text
Vision
  ↓
Gesture
  ↓
Interaction
  ↓
Application Logic
  ↓
Visualization
```

---

## VisiCLI Relationship

VisiGUI is not a separate implementation of the project analysis system.

It builds on the concepts and core logic of:

**VisiCLI**

https://github.com/AdolfMacro/VisiCLI

VisiCLI focuses on the command-line side of project analysis and interaction.

VisiGUI adds a graphical layer on top of that foundation.

```text
                    VisiCLI
                       │
             Core Analysis Logic
                       │
              ProjectGraph
                       │
              Gesture / Control
                       │
                       ▼
                    VisiGUI
                 ┌─────┴─────┐
                 │           │
                 ▼           ▼
          Project Explorer   3D World
```

This allows the analysis logic and graphical presentation to remain separate.

---

## Safety Model

The target project is analyzed statically.

VisiGUI does **not** need to import or execute the target project in order to construct its project graph.

This is important because the project being inspected may contain:

* Arbitrary application code
* Network operations
* External dependencies
* Side effects
* Startup logic

The analysis layer therefore works with source structure rather than executing the inspected application.

---

## Project Navigation

The project graph allows VisiGUI to represent a Python project as a hierarchy.

```text
Project
│
├── Package
│   ├── Module
│   │   ├── Class
│   │   │   ├── Method
│   │   │   └── Method
│   │   │
│   │   └── Function
│   │
│   └── Module
│
└── Module
```

This structure provides the foundation for graphical project exploration.

---

## Interaction Model

The interaction system follows a layered approach.

### 1. Detection

The camera provides the input frame.

### 2. Recognition

The vision layer detects the hand and identifies a gesture.

### 3. Interpretation

The gesture is converted into a semantic event.

### 4. Control

The controller determines the appropriate action based on the current context.

### 5. Visualization

The selected action changes either the project explorer or the generative 3D environment.

```text
Camera
  ↓
Detection
  ↓
Gesture
  ↓
Semantic Event
  ↓
Controller
  ↓
Action
  ↓
Visual Output
```

---

## Generative World Interaction

The generative environment is designed as an independent graphical mode.

The interaction can be summarized as:

```text
Right Hand
   │
   ├── Zoom
   └── Zoom Out

Left Hand
   │
   └── Navigation / Movement
             │
             ▼
      Generative 3D World
```

The exact interaction rules can evolve without changing the underlying project-analysis architecture.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AdolfMacro/VisiGUI.git
cd VisiGUI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python3 main.py
```

---

## Usage

Run VisiGUI directly:

```bash
python3 main.py
```

The application starts with its graphical interface and provides access to the available project exploration and interactive visualization features.

The project is designed to work without requiring additional command-line arguments for the default experience.

---

## Project Structure

```text
VisiGUI/
│
├── src/
│   ├── core/
│   ├── engine3d/
│   ├── camera/
│   ├── vision/
│   ├── gesture/
│   ├── interaction/
│   ├── simulation/
│   ├── terminal/
│   └── world/
│
├── screenshots/
│   └── SC01.png
│
├── tests/
│
├── docs/
│
├── requirements.txt
└── README.md
```

The exact internal structure may evolve as the project develops.

---

## Design Principles

VisiGUI follows several core design principles:

### Separation of Concerns

Vision, gesture recognition, interaction, project analysis, and visualization are kept as separate layers.

### Semantic Interaction

A gesture should represent intent rather than directly triggering an implementation-specific action.

### Static Analysis

Target projects should be inspected without requiring their execution.

### Independent Visualization

The project explorer and generative 3D world should remain independent visual systems.

### Extensibility

New gestures, interaction modes, visual components, and analysis features should be possible without rewriting the entire application.

---

## Current Status

VisiGUI is an active development project.

The architecture is being built around:

* Static Python project analysis
* Project graph generation
* Gesture recognition
* Context-aware interaction
* Graphical project exploration
* Generative 3D visualization
* Camera-based interaction

Some parts of the system are still under development.

---

## Related Projects

### VisiCLI

Command-line project exploration and analysis.

https://github.com/AdolfMacro/VisiCLI

### VisiGUI

Graphical project exploration and interactive generative visualization.

https://github.com/AdolfMacro/VisiGUI

---

## Screenshot

<p align="center">
  <img src="https://raw.githubusercontent.com/AdolfMacro/VisiGUI/main/screenshots/SC01.png" alt="VisiGUI" width="1000">
</p>

---

## License

This project is licensed under the MIT License.

---

## Author

**Mani.k (AdolfMacro)**

* GitHub: https://github.com/AdolfMacro
* Website: https://adolfmacro.github.io/
* Email: [m4nikamran@gmail.com](mailto:m4nikamran@gmail.com)
