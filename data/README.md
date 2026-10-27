# Synthetic Spotify-like dataset

`synthetic_spotify_tracks.csv` contains 600 randomly generated rows and 13 columns. It is fabricated for pipeline smoke-testing; it is not downloaded from Spotify and does not represent real songs, artists, or Spotify popularity.

- Target: `track_popularity` (synthetic score from 0–100)
- Default positive class: `track_popularity >= 70`
- Seed: 20261027
- Use: verify CSV loading, preprocessing, model training, evaluation, and Streamlit prediction flow.
- Do not cite the resulting metrics as real-world performance or as a reproduction of the resume's original dataset result.
