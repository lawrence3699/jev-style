# Snake: licences and sources

- All code in this folder (`snake-core.js`, `snake.js`, `snake.css`, `index.html`), the Python twin
  (`the original project`) and the tests (`its tests`) were written for this project.
  No third-party game code, sprites, sounds, fonts, libraries or CDNs are used; the board is drawn on a
  canvas with the site's own colour tokens from `/common.css`.
- The seeded random generator is a plain xorshift32 (Marsaglia's published shift-register method),
  implemented here from the description.
- No data files: every game is generated from its seed.
