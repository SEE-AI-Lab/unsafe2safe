---
name: SafeFrame
description: A clear, ticket-wallet-inspired photo editing flow.
colors:
  carrier-navy: "#0d1b3d"
  jet-red: "#d7262d"
  stock-white: "#f2f2f0"
  coupon-white: "#ffffff"
  carbon-purple: "#4b2e83"
  ink: "#152442"
  muted: "#58677d"
  rule: "#d5dbe3"
  photo-wash: "#e8edf2"
  success-green: "#27634c"
  focus-lilac: "#b676d1"
  brand-coral: "#ef5449"
  step-rule: "#cbd2db"
  step-outline: "#aab6c5"
  selected-wash: "#fffafa"
  image-matte: "#e1e5e9"
  photo-hover-wash: "#edf0f3"
  choice-hover-rule: "#8796ac"
  button-hover: "#b91f26"
  disabled-gray: "#aeb7c3"
  error-red: "#a51e24"
typography:
  display:
    fontFamily: "Barlow Condensed, Impact, sans-serif"
    fontSize: "36px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.015em"
  body:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    letterSpacing: "0.07em"
rounded:
  panel: "4px"
  control: "3px"
  circular: "50%"
spacing:
  choice-gap: "7px"
  control-inset: "11px"
  panel-inset: "19px"
  workspace-gap: "18px"
  page-gutter: "24px"
components:
  panel:
    backgroundColor: "{colors.coupon-white}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel-inset}"
  button-primary:
    backgroundColor: "{colors.jet-red}"
    textColor: "{colors.coupon-white}"
    rounded: "{rounded.control}"
    padding: "11px 17px"
    height: "42px"
  button-primary-hover:
    backgroundColor: "{colors.button-hover}"
  edit-choice-selected:
    backgroundColor: "{colors.selected-wash}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
---

# Design System: SafeFrame

## Overview

**Creative North Star: “The Ticket Wallet.”**

SafeFrame turns one photo edit into a short, legible trip: choose a source, select one treatment, and collect the finished image. The ticket-wallet reference contributes its navy and red identity, condensed display type, tab-like labels, crisp dividers, and a single three-step progress signature. The interface stays an ordinary photo tool; travel language is an organizing cue, not a layer of invented travel facts.

The two-panel workspace keeps the source and its compact edit controls together on the left while the result remains visible on the right from the first desktop view. The same hierarchy collapses into one mobile column, with the source and edit choices before the result. Privacy and creative modes share the same clear selection and action pattern.

**Key Characteristics:**

- A short, ordered Choose → Edit → Ready journey.
- A restrained carrier palette with red reserved for active progress and primary action.
- Standard, legible photo controls inside a crisp, low-depth layout.

## Colors

The palette pairs a deep carrier navy and warm stock white with a precise jet-red action accent; purple marks processing and coral distinguishes the brand mark.

### Primary
- **Carrier Navy** (`{colors.carrier-navy}`): Brand bar, major headings, current progress, and key text within photo controls.
- **Jet Red** (`{colors.jet-red}`): Primary action, current or completed step markers, and selected edit outline.

### Secondary
- **Carbon Purple** (`{colors.carbon-purple}`): Processing emphasis and the compact category labels attached to edit choices.

### Neutral
- **Stock White** (`{colors.stock-white}`): Main page canvas.
- **Coupon White** (`{colors.coupon-white}`): Header text and panel surfaces.
- **Ink** (`{colors.ink}`): Default body text and edit-choice text.
- **Muted Slate** (`{colors.muted}`): Supporting copy, helper text, and inactive status text.
- **Rule Gray** (`{colors.rule}`): Panel and choice outlines and the progress-bar baseline.
- **Photo Wash** (`{colors.photo-wash}`): Empty photo and result stage backgrounds.
- **Success Green** (`{colors.success-green}`): Successful result state.
- **Focus Lilac** (`{colors.focus-lilac}`): Visible keyboard focus outline.
- **Brand Coral** (`{colors.brand-coral}`): SafeFrame wordmark accent.
- **Step Rule** (`{colors.step-rule}`) and **Step Outline** (`{colors.step-outline}`): Inactive progress connectors and number outlines.
- **Selected Wash** (`{colors.selected-wash}`): Selected edit background.
- **Image Matte** (`{colors.image-matte}`): Background behind the contained source image.
- **Photo Hover Wash** (`{colors.photo-hover-wash}`) and **Choice Hover Rule** (`{colors.choice-hover-rule}`): Photo-picker and edit-choice hover feedback.
- **Button Hover** (`{colors.button-hover}`): Darker primary-button hover state.
- **Disabled Gray** (`{colors.disabled-gray}`): Disabled primary action.
- **Error Red** (`{colors.error-red}`): Error status text.

## Typography

**Display Font:** Barlow Condensed (with Impact, sans-serif fallback)  
**Body Font:** DM Sans (with sans-serif fallback)

**Character:** Barlow Condensed gives the main heading and ticket-like category labels a compact, recognizable voice. DM Sans keeps instructions, choices, and controls familiar and readable.

### Hierarchy
- **Display** (700, 36px, line-height 1, letter-spacing 0.015em): Main screen heading; it reduces to 32px on narrow phones.
- **Title** (700, 14px): Photo and result panel headings, and the edit-mode prompt.
- **Body** (400, 15px, line-height 1.45): Default interface copy.
- **Supporting** (400, 11–13px): Choice descriptions, helper text, status details, and the header tagline.
- **Label** (700, 11–12px, tracked, uppercase where appropriate): State labels and the Barlow Condensed edit-category tags.

## Layout

The desktop header and progress row align to a 1316px content measure, while the page is capped at 1440px. The workspace uses two panels in a 1.15:0.85 grid with an 18px gap; the left panel combines source selection, the three edit modes, and the action, while the right panel holds the result state. Keep the result panel visible beside the source in the initial desktop view.

At 850px and below, stack the panels in document order and reduce the top inset. At 560px and below, tighten the page gutters, panel padding, and image stages; stack the action note and full-width button. The step markers stay horizontal. Preserve the semantic order of source, edit choice, action, then result.

## Elevation & Depth

The system is flat at rest and uses tonal changes, thin outlines, and the red selected state to distinguish controls and surfaces. The implementation has no box shadows. Reserve movement for the primary button’s subtle hover lift, and remove transitions when the user requests reduced motion.

## Shapes

Panels use a small 4px corner; buttons and edit choices use a tighter 3px corner. Photo stages remain square-cornered and outlined, keeping the image area neutral and practical. Radio marks and numbered progress markers are circular. Thin gray rules define the paper-like structure; avoid ornamental perforations or simulated ticket edges.

## Components

### Buttons
- **Character:** Direct and confident, with the red primary action as the clearest next step.
- **Shape:** Tight 3px corner.
- **Primary:** Jet-red fill, white text, 42px minimum height, and 11px 17px padding.
- **Hover / Focus:** Darken on hover with a 1px upward shift; keyboard focus uses a 3px lilac outline with 3px offset.
- **Disabled:** Gray fill, disabled cursor, and no hover lift.

### Cards / Containers
- **Character:** Crisp white working panels rather than decorative cards.
- **Corner Style:** 4px.
- **Background:** Coupon-white surface over the stock-white page.
- **Shadow Strategy:** None; a 1px rule supplies the edge.
- **Internal Padding:** 19px on desktop and 13px on narrow phones.

### Inputs / Fields
- **Style:** The image picker is a full photo-stage label with a dashed outline and a transparent file input filling the stage; it accepts image files and requests the rear camera on mobile.
- **Focus:** A visible lilac outline is applied to focusable controls and the picker label.
- **Error / Disabled:** The primary action is disabled until a photo is selected; failures use red status copy and a retry-ready action.

### Navigation
- **Style:** A white horizontal three-step progress row below the navy brand bar. Each step uses a numbered circle, a short label, and a thin connector.
- **States:** Red-filled circles mark current and completed steps; the current label is navy and bold. Success advances the sequence to its completed state.
- **Mobile:** Keep all three steps in one compact horizontal row.

### Edit Choices
- **Character:** Three compact, clearly named treatments, with a small category tag that carries the ticket-wallet reference.
- **Shape:** Tight 3px corners, a circular radio mark, and a fine neutral border.
- **Selected:** Red 2px outline, pale red-white wash, and a red radio center; selection is exposed through `aria-pressed`.
- **Hover:** Neutral border darkens slightly without changing the selected state.

### Photo and Result Stages
- **Character:** Neutral image frames make the original and generated image the visual focus.
- **Photo Stage:** Fixed 270px desktop height with a dashed border; on narrow phones it is 230px.
- **Result Stage:** Fills the result panel, with a 400px minimum height on desktop and 270px on narrow phones. Images use `object-fit: contain` so the complete image remains visible.
- **States:** Empty and processing text stays centered. Purple distinguishes active processing; a successful result receives a green status and a text download link.

## Do's and Don'ts

### Do:
- **Do** keep the desktop source-and-controls panel beside a visible result panel from the first view.
- **Do** keep the three edit modes distinct and selectable before generation.
- **Do** use the three-step progress marker to show photo, edit, and result status.
- **Do** preserve the original/result distinction and provide the download link only when a result is ready.
- **Do** retain the visible focus treatment and reduced-motion behavior.

### Don't:
- **Don't** turn the editor into a feed, gallery, or stack of decorative upload cards.
- **Don't** add invented flight information or travel facts to the ticket-wallet metaphor.
- **Don't** obscure the selected photo, edit choice, generation state, or result download behind decorative travel details.
