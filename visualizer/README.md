## Cluster Visualizer

Interactive React + Vite dashboard for exploring `data/landmarks/all_segments_clustered_with_xy.json`. It groups frames per cluster/segment and lets you jump straight into the exported clip.

### Getting started

```bash
cd visualizer
npm install
npm run dev
```

- Uses Vite with React + TypeScript. Vite currently requires Node `^20.19.0` or `>= 22.12.0` (you will see a warning on older builds).
- The app expects the cluster JSON and the exported `data/segments/*.mp4` files to be reachable at runtime. The repo includes symlinks:
  - `visualizer/public/all_segments_clustered_with_xy.json → ../data/landmarks/all_segments_clustered_with_xy.json`
  - `visualizer/public/segments → ../data/segments`

If your OS or deployment target cannot follow those symlinks, copy the assets manually instead:

```bash
cp data/landmarks/all_segments_clustered_with_xy.json visualizer/public/
cp -R data/segments visualizer/public/
```

### Features

- Summary cards for cluster/segment coverage and an editable FPS control (defaults to 25).
- Cluster picker that surfaces counts/percentages (noise cluster is treated specially).
- 3D embedding scatter plot (PCA space) for the selected cluster; hover any point to capture the matching video frame inline.
- Segment drill-down showing every frame assigned to the selected cluster.
- Inline video player that loads `segment_###.mp4` and seeks to the highlighted frame (frame index ÷ FPS).

### Production build

```
npm run build
```
