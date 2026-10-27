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
  focus-lilac: "#b676d1"
  brand-coral: "#ef5449"
  selected-wash: "#fffafa"
  image-matte: "#e1e5e9"
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
    fontSize: "13px"
    fontWeight: 700
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
    padding: "10px 16px"
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

SafeFrame turns one photo edit into a short, legible trip: choose a source, select one treatment, and collect the finished image. The ticket-wallet reference contributes its navy and red identity, condensed display type, crisp dividers, and a compact journey. Travel language is an organizing cue, not a layer of invented travel facts.

The workspace pairs the source and result photos across the page, then places the caption comparison and edit controls in balanced columns directly below. This keeps both images aligned and gives each paper stage room to read without squeezing Stage 2 into a narrow sidebar. On mobile, the hierarchy collapses into one column. Upload and camera capture are separate, equally clear ways to provide a photo; camera preview and capture controls appear inline only after the visitor starts the camera.

**Key Characteristics:**

- A short, ordered choose → edit → collect flow with a compact, live progress tracker.
- A restrained carrier palette with red reserved for the primary action and selected edit.
- Standard, legible photo controls inside a crisp, low-depth layout.

## Colors

The palette pairs deep carrier navy and warm stock white with a precise jet-red action accent; purple distinguishes active processing.

### Primary
- **Carrier Navy** (`{colors.carrier-navy}`): Brand bar, main heading, and key text within photo controls.
- **Jet Red** (`{colors.jet-red}`): Primary action and selected edit outline.

### Secondary
- **Carbon Purple** (`{colors.carbon-purple}`): Processing emphasis.

### Neutral
- **Stock White** (`{colors.stock-white}`): Main page canvas.
- **Coupon White** (`{colors.coupon-white}`): Header text and panel surfaces.
- **Ink** (`{colors.ink}`): Default body text and edit-choice text.
- **Muted Slate** (`{colors.muted}`): Supporting copy and helper text.
- **Rule Gray** (`{colors.rule}`): Panel and choice outlines.
- **Photo Wash** (`{colors.photo-wash}`): Empty photo and result stage backgrounds.
- **Focus Lilac** (`{colors.focus-lilac}`): Visible keyboard focus outline.
- **Brand Coral** (`{colors.brand-coral}`): SafeFrame wordmark accent.
- **Selected Wash** (`{colors.selected-wash}`): Selected edit background.
- **Image Matte** (`{colors.image-matte}`): Background behind the contained source image.
- **Choice Hover Rule** (`{colors.choice-hover-rule}`): Edit-choice hover feedback.
- **Button Hover** (`{colors.button-hover}`): Darker primary-button hover state.
- **Disabled Gray** (`{colors.disabled-gray}`): Disabled primary action.
- **Error Red** (`{colors.error-red}`): Camera and edit error text.

## Typography

**Display Font:** Barlow Condensed (with Impact, sans-serif fallback)  
**Body Font:** DM Sans (with sans-serif fallback)

**Character:** Barlow Condensed gives the main heading a compact, recognizable voice. DM Sans keeps instructions, choices, and controls familiar and readable.

### Hierarchy
- **Display** (700, 36px, line-height 1, letter-spacing 0.015em): Main screen heading; it reduces to 32px on narrow phones.
- **Title** (700, 14px): Photo and result panel headings, and the edit-mode prompt.
- **Body** (400, 15px, line-height 1.45): Default interface copy and captions.
- **Supporting** (400, 14px): Privacy note, choice descriptions, and generated instruction.
- **Control label** (700, 14–15px): Button labels, progress labels, and section headings.

## Layout

The page is capped at 1920px. The workspace pairs source and result panels in an even two-column grid, then uses a full-width workflow panel with balanced Stage 1 and Stage 2 columns. Stage 1 compares private and public captions side by side; Stage 2 shows three compact, equal-width edit choices above the generated instruction and primary action. Keep source and result visible together in the initial desktop view.

At 850px and below, stack the panels in document order and reduce the top inset. At 560px and below, tighten the page gutters, panel padding, and image stages; make the upload and camera buttons share the row and the primary action full width. Preserve the order of source, edit choice, action, then result.

On desktop viewports shorter than 1050px, compact the paper header and progress row and cap the paired photo stages at 44vh (300–460px). Keep both images fully visible with `object-fit: contain`; the workflow may continue below the fold and remains available by scrolling.

## Elevation & Depth

The system is flat at rest and uses tonal changes and thin outlines to distinguish controls and surfaces. The implementation has no box shadows. Reserve movement for the primary button’s subtle hover lift, and remove transitions when the user requests reduced motion.

## Shapes

Panels use a small 4px corner; buttons and edit choices use a tighter 3px corner. Photo stages remain square-cornered and outlined, keeping the image area neutral and practical. Radio marks are circular. Thin gray rules define the paper-like structure; avoid ornamental perforations or simulated ticket edges.

## Components

### Buttons
- **Character:** Direct and confident, with the red primary action as the clearest next step.
- **Shape:** Tight 3px corner.
- **Primary:** Jet-red fill, white text, 42px minimum height, and 10px 16px padding.
- **Secondary:** White fill, navy text, neutral border, and the same compact control shape for upload and camera actions.
- **Hover / Focus:** Darken and lift the primary button by 1px on hover; keyboard focus uses a 3px lilac outline with 3px offset.
- **Disabled:** Gray fill, disabled cursor, and no hover lift.

### Cards / Containers
- **Character:** Crisp white working panels rather than decorative cards.
- **Corner Style:** 4px.
- **Background:** Coupon-white surface over the stock-white page.
- **Shadow Strategy:** None; a 1px rule supplies the edge.
- **Internal Padding:** 19px on desktop and 13px on narrow phones.

### Inputs / Fields
- **Photo selection:** The dashed photo stage shows a neutral prompt. Separate “Upload a photo” and “Take a photo” buttons sit directly below it; upload opens the image picker.
- **Camera:** “Take a photo” requests browser camera permission on first use, then replaces the photo actions with an inline live preview and Capture photo / Cancel controls. Capture selects a still image; Cancel stops the camera and returns to the two actions. Camera errors appear as concise red helper text.
- **Focus:** A visible lilac outline is applied to focusable controls.
- **Error / Disabled:** The primary action is disabled until a photo is selected; failures use red status copy and a retry-ready action.

### Edit Choices
- **Character:** Three compact, clearly named treatments, without category tags or extra status labels.
- **Shape:** Tight 3px corners, a circular radio mark, and a fine neutral border.
- **Selected:** Red 2px outline, pale red-white wash, and a red radio center; selection is exposed through `aria-pressed`.
- **Hover:** Neutral border darkens slightly without changing the selected state.

### Photo and Result Stages
- **Character:** Neutral image frames make the original and generated image the visual focus.
- **Photo and Result Stages:** Use the same aspect ratio and align side by side on desktop. Images use `object-fit: contain` so the complete image remains visible.
- **States:** The result placeholder communicates the next state; processing is emphasized in purple. A successful result exposes a text download link.

## Do's and Don'ts

### Do:
- **Do** keep the desktop source-and-controls panel beside a visible result panel from the first view.
- **Do** keep the three edit modes distinct and selectable before generation.
- **Do** give upload and camera capture separate, equally visible actions.
- **Do** preserve the original/result distinction and provide the download link only when a result is ready.
- **Do** retain the visible focus treatment and reduced-motion behavior.

### Don't:
- **Don't** turn the editor into a feed, gallery, or stack of decorative upload cards.
- **Don't** add invented flight information or travel facts to the ticket-wallet metaphor.
- **Don't** reintroduce progress navigation, tag labels, taglines, or redundant status captions that compete with the photo and edit controls.
