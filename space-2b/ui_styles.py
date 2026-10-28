"""JevStyle demo presentation for Gradio 6.28.x.

Import CSS and make_theme; pass make_theme(gr) and CSS to demo.launch().
This module has no model, network, Gradio import, or application side effects.
Layout and state contracts are in ui_integration.txt next to this file.
"""

THEME_KWARGS = {
    "primary_hue": "violet",
    "secondary_hue": "blue",
    "neutral_hue": "slate",
    "spacing_size": "sm",
    "radius_size": "lg",
    "text_size": "md",
    # Use an installed font or the OS fallback; do not block rendering on a font CDN.
    "font": ["Figtree", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    "font_mono": ["SFMono-Regular", "Consolas", "ui-monospace", "monospace"],
}

THEME_TOKENS = {
    "body_background_fill": "#ffffff",
    "body_background_fill_dark": "#0d0d12",
    "body_text_color": "#13131b",
    "body_text_color_dark": "#f2f2f7",
    "body_text_color_subdued": "#62626f",
    "body_text_color_subdued_dark": "#aaaab9",
    "background_fill_primary": "#ffffff",
    "background_fill_primary_dark": "#14141c",
    "background_fill_secondary": "#f7f7fa",
    "background_fill_secondary_dark": "#1b1b25",
    "border_color_primary": "#e5e5ed",
    "border_color_primary_dark": "#30303e",
    "border_color_accent": "#6352d8",
    "border_color_accent_dark": "#a79af5",
    "link_text_color": "#5542c6",
    "link_text_color_dark": "#b4a8ff",
    "link_text_color_hover": "#372497",
    "link_text_color_hover_dark": "#d0c7ff",
    "block_background_fill": "#ffffff",
    "block_background_fill_dark": "#14141c",
    "block_border_color": "#e5e5ed",
    "block_border_color_dark": "#30303e",
    "block_border_width": "1px",
    "block_radius": "12px",
    "block_shadow": "none",
    "block_shadow_dark": "none",
    "block_label_text_size": "13px",
    "block_label_text_weight": "500",
    "block_title_text_size": "13px",
    "block_title_text_weight": "500",
    "input_background_fill": "#fbfbfd",
    "input_background_fill_dark": "#111119",
    "input_border_color": "#dedee8",
    "input_border_color_dark": "#343443",
    "input_border_color_focus": "#6352d8",
    "input_border_color_focus_dark": "#a79af5",
    "input_radius": "9px",
    "input_shadow": "none",
    "input_shadow_dark": "none",
    "input_shadow_focus": "0 0 0 3px rgba(99,82,216,.12)",
    "input_shadow_focus_dark": "0 0 0 3px rgba(167,154,245,.18)",
    "checkbox_label_background_fill": "#ffffff",
    "checkbox_label_background_fill_dark": "#14141c",
    "checkbox_label_background_fill_hover": "#f7f5ff",
    "checkbox_label_background_fill_hover_dark": "#211c36",
    "checkbox_label_background_fill_selected": "#f0ecff",
    "checkbox_label_background_fill_selected_dark": "#282140",
    "checkbox_label_border_color": "#e5e5ed",
    "checkbox_label_border_color_dark": "#343443",
    "checkbox_label_border_color_selected": "#8a77e8",
    "checkbox_label_border_color_selected_dark": "#a79af5",
    "checkbox_label_text_color_selected": "#44309e",
    "checkbox_label_text_color_selected_dark": "#d2c8ff",
    "checkbox_label_shadow": "none",
    "checkbox_label_shadow_dark": "none",
    "checkbox_label_padding": "12px 16px",
    "button_primary_background_fill": "#17171e",
    "button_primary_background_fill_dark": "#ebe7ff",
    "button_primary_background_fill_hover": "#4130a0",
    "button_primary_background_fill_hover_dark": "#d4caff",
    "button_primary_border_color": "#17171e",
    "button_primary_border_color_dark": "#ebe7ff",
    "button_primary_text_color": "#ffffff",
    "button_primary_text_color_dark": "#201646",
    "button_primary_shadow": "none",
    "button_primary_shadow_dark": "none",
    "button_secondary_background_fill": "#ffffff",
    "button_secondary_background_fill_dark": "#171720",
    "button_secondary_background_fill_hover": "#f7f5ff",
    "button_secondary_background_fill_hover_dark": "#252034",
    "button_secondary_border_color": "#dedee8",
    "button_secondary_border_color_dark": "#3c3c4c",
    "button_secondary_text_color": "#3f3f4b",
    "button_secondary_text_color_dark": "#dfdce9",
    "button_secondary_shadow": "none",
    "button_secondary_shadow_dark": "none",
}


def make_theme(gr):
    """Construct a Gradio theme without requiring Gradio when importing this file."""
    return gr.themes.Default(**THEME_KWARGS).set(**THEME_TOKENS)


CSS = r"""
/* Stable hooks only: assigned IDs/classes, native controls, our own status HTML.
   No generated Svelte classes, positional selectors, or hidden Gradio controls. */
:is(.gradio-container, #jev-app, .jev-app) {
  --jev-ink: var(--body-text-color, #13131b);
  --jev-muted: var(--body-text-color-subdued, #62626f);
  --jev-bg: var(--body-background-fill, #fff);
  --jev-panel: var(--background-fill-primary, #fff);
  --jev-soft: var(--background-fill-secondary, #f7f7fa);
  --jev-line: var(--border-color-primary, #e5e5ed);
  --jev-accent: var(--link-text-color, #5542c6);
  max-width: 1160px !important;
  margin-inline: auto;
  padding: 28px 28px 22px !important;
  color: var(--jev-ink);
  background: var(--jev-bg);
}

/* Short introduction; the controls stay in the first screen on a laptop. */
#jev-hero { padding: 0 0 10px; border: 0; background: transparent; }
#jev-hero h1 {
  max-width: 860px;
  margin: 8px 0 13px;
  font-size: clamp(28px, 3.5vw, 43px);
  line-height: 1.12;
  font-weight: 600;
  letter-spacing: -.045em;
  color: var(--jev-ink);
}
#jev-hero p { max-width: 760px; margin: 0 0 10px; color: var(--jev-muted); font-size: 16px; line-height: 1.6; }
.jev-eyebrow { color: var(--jev-accent); font-size: 11px; letter-spacing: .09em; font-weight: 600; text-transform: uppercase; }
.jev-toplinks { display: flex; flex-wrap: wrap; gap: 8px 19px; margin: 15px 0 0; font-size: 12px; }
.jev-toplinks a { color: var(--jev-muted); text-decoration: none; }
.jev-toplinks a:hover { color: var(--jev-accent); text-decoration: underline; text-underline-offset: 3px; }

/* Native Radio option labels remain keyboard accessible and show checked state. */
#jev-task-picker { padding: 16px 0 17px; border: 0; background: transparent; }
#jev-task-picker label:has(input[type="radio"]) {
  min-height: 47px;
  align-items: center;
  border-radius: 10px;
  font-weight: 500;
  transition: background-color .15s, border-color .15s;
}
#jev-task-picker input[type="radio"] { accent-color: var(--jev-accent); }
#jev-task-picker label:has(input[type="radio"]:checked) { box-shadow: inset 0 0 0 1px var(--jev-accent); }
#jev-task-picker label:focus-within { outline: 2px solid var(--jev-accent); outline-offset: 3px; }

/* Use Gradio Row/Column scale and min_width for layout; CSS improves spacing. */
:is(#jev-workspace, .jev-workspace) { gap: 24px !important; align-items: flex-start; }
:is(#jev-input-panel, #jev-output-panel, .jev-input-panel, .jev-output-panel) { min-width: 0; gap: 14px !important; }
.jev-section-heading h2, .jev-section-heading h3 {
  margin: 0 0 3px;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -.015em;
  color: var(--jev-ink);
}
.jev-section-heading p { margin: 0; color: var(--jev-muted); font-size: 12px; line-height: 1.5; }
.jev-field textarea { line-height: 1.6; font-size: 14px; }
.jev-advanced { background: var(--jev-soft); border-color: var(--jev-line); }
.jev-advanced summary { font-size: 13px; font-weight: 500; }
.jev-meta p { margin: 0; color: var(--jev-muted); font-size: 12px; line-height: 1.6; }
.jev-code { font-size: 12px; }

/* The same result box is used for ready, queued, live, cached, and error states.
   State text is always supplied by the app; CSS never invents a live/cached label. */
#jev-status, .jev-status-host { border: 0; padding: 0; background: transparent; }
.jev-status {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid var(--jev-line);
  border-radius: 12px;
  background: var(--jev-soft);
  color: var(--jev-ink);
}
.jev-status-body { min-width: 0; flex: 1; }
.jev-status-title { margin: 0; font-size: 13px; font-weight: 600; line-height: 1.5; }
.jev-status-description { margin: 4px 0 0; color: var(--jev-muted); font-size: 12px; line-height: 1.55; }
.jev-state-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: 0 0 auto;
  border: 1px solid var(--jev-line);
  border-radius: 999px;
  padding: 4px 8px;
  background: var(--jev-panel);
  color: var(--jev-muted);
  font-size: 10px;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: .025em;
  white-space: nowrap;
}
.jev-state-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.jev-status[data-state="live"] .jev-state-badge { color: #176849; border-color: #b2d8c7; background: #eef9f3; }
.jev-status[data-state="cached"] .jev-state-badge { color: #57419e; border-color: #cfc2ed; background: #f3effb; }
.jev-status[data-state="queued"] .jev-state-badge,
.jev-status[data-state="running"] .jev-state-badge { color: #355ea7; border-color: #b9cdee; background: #eef4ff; }
.jev-status[data-state="error"] { border-color: #e6c7a0; }
.jev-status[data-state="error"] .jev-state-badge { color: #855016; border-color: #e6c7a0; background: #fff7eb; }
.jev-status[data-state="stale"] .jev-state-badge { color: var(--jev-muted); border-style: dashed; }

/* Results never sit behind a full-panel busy overlay. The app must keep recovery
   actions outside disabled inputs and use queue=False for saved-example events. */
#jev-probabilities, .jev-result { min-height: 194px; border-radius: 12px; }
.jev-empty-result {
  min-height: 192px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  padding: 24px;
  border: 1px dashed var(--jev-line);
  border-radius: 12px;
  background: var(--jev-panel);
}
.jev-empty-result strong { font-size: 19px; line-height: 1.35; font-weight: 500; letter-spacing: -.025em; }
.jev-empty-result p { max-width: 32ch; margin: 9px 0 0; color: var(--jev-muted); font-size: 13px; line-height: 1.6; }
.jev-many table { width: 100%; font-size: 12px; }
.jev-many td, .jev-many th { overflow-wrap: anywhere; vertical-align: top; }

#jev-recovery, .jev-recovery {
  padding: 16px;
  border: 1px solid var(--jev-line);
  border-radius: 12px;
  background: var(--jev-soft);
}
.jev-recovery h3 { margin: 0 0 7px; font-size: 14px; line-height: 1.4; font-weight: 600; }
.jev-recovery p { margin: 0; color: var(--jev-muted); font-size: 13px; line-height: 1.6; }
.jev-recovery a { color: var(--jev-accent); text-underline-offset: 3px; }
.jev-recovery-hint { margin: 10px 0 0 !important; font-size: 12px !important; }

/* Permanent escape routes have no animation and remain visually normal while
   a live request queues/runs/fails. Never change pointer-events to override state. */
#jev-always-actions, .jev-always-actions { gap: 10px !important; align-items: stretch; }
:is(#jev-run, #jev-saved, #jev-local, .jev-action) { min-height: 44px; border-radius: 10px; font-size: 13px; font-weight: 600; }
#jev-run { margin-top: 2px; }
#jev-saved { white-space: normal; }
#jev-local { text-decoration: none; }
.jev-local-link { display: flex; align-items: center; justify-content: center; gap: 6px; min-height: 44px; padding: 9px 15px; border: 1px solid var(--jev-line); border-radius: 10px; background: var(--jev-panel); color: var(--jev-ink); text-decoration: none; font-size: 13px; font-weight: 600; }
.jev-local-link:hover { border-color: var(--jev-accent); color: var(--jev-accent); background: var(--jev-soft); }
.jev-action-help { margin: 0; color: var(--jev-muted); font-size: 11px; line-height: 1.5; }

.jev-example-library { margin-top: 8px; }
.jev-example-library summary { font-size: 13px; }
.jev-foot { padding-top: 20px; border-top: 1px solid var(--jev-line); margin-top: 22px; }
.jev-foot p { color: var(--jev-muted); font-size: 11px; line-height: 1.7; }
:is(.gradio-container, #jev-app, .jev-app) :is(a, button, input, textarea, summary):focus-visible { outline: 2px solid var(--jev-accent); outline-offset: 3px; }

/* Respect HF/Gradio dark mode without rewriting its class names or preference. */
.dark .jev-status[data-state="live"] .jev-state-badge { color: #a9e1c6; background: #17352b; border-color: #356652; }
.dark .jev-status[data-state="cached"] .jev-state-badge { color: #d2c2ff; background: #302642; border-color: #5c4b7d; }
.dark .jev-status[data-state="queued"] .jev-state-badge,
.dark .jev-status[data-state="running"] .jev-state-badge { color: #b8cff6; background: #1e304d; border-color: #3f5d8a; }
.dark .jev-status[data-state="error"] { border-color: #755532; }
.dark .jev-status[data-state="error"] .jev-state-badge { color: #efcd9f; background: #3c2b1a; border-color: #755532; }

@media (min-width: 901px) {
  #jev-output-panel { position: sticky; top: 18px; }
}
@media (max-width: 900px) {
  :is(.gradio-container, #jev-app, .jev-app) { padding: 22px 20px 20px !important; }
  :is(#jev-workspace, .jev-workspace) { gap: 19px !important; }
  #jev-output-panel { position: static; }
}
@media (max-width: 700px) {
  :is(.gradio-container, #jev-app, .jev-app) { padding: 16px 12px !important; }
  #jev-hero h1 { font-size: 29px; margin-top: 5px; }
  #jev-hero p { font-size: 14px; line-height: 1.6; }
  #jev-task-picker { padding-block: 10px 13px; }
  #jev-task-picker label:has(input[type="radio"]) { min-height: 44px; padding: 10px 12px; font-size: 12px; }
  :is(#jev-workspace, .jev-workspace) { flex-direction: column; }
  :is(#jev-input-panel, #jev-output-panel, .jev-input-panel, .jev-output-panel) { width: 100% !important; min-width: 0 !important; }
  .jev-status { padding: 12px; gap: 9px; }
  .jev-status-description { font-size: 12px; }
  .jev-state-badge { font-size: 9px; padding: 4px 6px; }
  .jev-empty-result { min-height: 145px; padding: 20px; }
  #jev-probabilities, .jev-result { min-height: 150px; }
  .jev-field textarea { font-size: 16px; }
  .jev-toplinks { gap: 7px 15px; font-size: 11px; }
  .jev-foot { margin-top: 14px; }
}
@media (prefers-reduced-motion: reduce) {
  #jev-task-picker label:has(input[type="radio"]) { transition: none; }
}
"""

CSS += r"""
#jev-output-panel { position: static; }
.jev-wait-clock { margin: 7px 0 0; color: var(--jev-muted); font-size: 12px; }
.jev-always-actions { display: grid; grid-template-columns: 1fr 1fr; margin-bottom: 8px; }
.submitted-request { font-size: 12px; color: var(--jev-muted); padding: 10px 0; }
.submitted-request pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 8px 0; }
.saved-examples { margin-top: 9px; }
.saved-examples h2 { font-size: 17px; font-weight: 600; margin: 12px 0 8px; letter-spacing: -.02em; }
.saved-disclaimer { font-size: 12px; line-height: 1.6; color: var(--jev-muted); margin-bottom: 14px; }
.saved-card { border: 1px solid var(--jev-line); border-radius: 11px; margin: 9px 0; background: var(--jev-panel); overflow: hidden; }
.saved-card > summary { padding: 14px; cursor: pointer; font-size: 13px; }
.saved-card-title { font-weight: 600; }
.saved-badge { display: block; color: var(--jev-accent); font-size: 10px; margin: 5px 0 0 16px; }
.saved-card[open] > summary { background: var(--jev-soft); border-bottom: 1px solid var(--jev-line); }
.saved-card-body { padding: 15px; }
.saved-card-body h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--jev-muted); margin: 0 0 10px; }
.saved-card-body h4 { font-size: 12px; margin: 12px 0 5px; }
.saved-card-body p,.saved-card-body pre { font-size: 12px; line-height: 1.6; margin: 0 0 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.saved-output { padding-top: 8px; }
.saved-probabilities { list-style: none; padding: 0; margin: 0; }
.saved-probability { margin: 12px 0; }
.saved-probability-label { display: flex; justify-content: space-between; gap: 10px; font-size: 13px; }
.saved-bar { display: block; width: 100%; height: 12px; accent-color: #6352d8; margin: 5px 0; }
.saved-probability-exact { color: var(--jev-muted); font-size: 10px; }
.saved-provenance { padding: 0 15px 15px; color: var(--jev-muted); font-size: 10px; }
.saved-provenance summary { cursor: pointer; font-size: 11px; }
.saved-provenance p { line-height: 1.6; margin: 9px 0; overflow-wrap: anywhere; }
.saved-receipt { max-width: 190px; width: 100%; height: auto; display: block; border-radius: 6px; margin: 10px 0; }
.saved-scope-note { font-size: 11px!important; padding: 9px; background: var(--jev-soft); border-radius: 6px; }
#jev-local-setup { margin-top: 30px; padding: 20px; border: 1px solid var(--jev-line); border-radius: 14px; }
#jev-local-setup h2 { font-size: 22px; font-weight: 600; letter-spacing: -.025em; }
@media(max-width:700px) { .jev-always-actions { grid-template-columns: 1fr; } #jev-local-setup { padding: 12px; } }
"""

CSS += r"""
.gradio-container { padding-top: 18px!important; }
#jev-hero h1 { font-size: clamp(29px, 3.4vw, 39px); margin: 8px 0 11px; }
#jev-hero p { font-size: 14px; margin-bottom: 6px; }
.jev-toplinks { margin-top: 10px; }
#jev-task-picker { padding: 8px 0 12px!important; background: transparent!important; border: 0!important; }
@media(max-width:700px) { #jev-hero h1 { font-size: 28px; } }
"""

CSS += r"""
#jev-probabilities { min-height: 130px; }
.saved-probability-exact { display: none; }
.saved-receipt img { width: 100%; height: auto; display: block; }
.saved-receipt figcaption { font-size: 10px; margin-top: 6px; color: var(--jev-muted); }
"""

CSS += r"""
.form:has(> #jev-task-picker) { background: transparent!important; border: 0!important; }
"""

CSS += r"""
/* Gradio's outer flex item needs an explicit shrinking width inside HF iframes. */
.gradio-container { width: 100%!important; min-width: 0!important; box-sizing: border-box; padding: 20px clamp(12px, 3vw, 28px)!important; }
.gradio-container .main { width: 100%; min-width: 0; box-sizing: border-box; padding: 0!important; }
.gradio-container .row,.gradio-container .column { min-width: 0; }
#jev-hero h1 { overflow-wrap: anywhere; }
"""
