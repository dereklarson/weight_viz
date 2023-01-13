# 

Transformer Visor (working title) allows for visual playback of the evolving model state during training.
It is forked from a favorite viz tool of mine, [Tensorflow Playground](https://github.com/tensorflow/playground).
It works in tandem with my [Transformer Lab](https://github.com/dereklarson/transformer-lab) repository, which is used to define
"experiments" (Sets of model and data configurations to run) and then dump the model checkpoint data as the frames for viewing.

I've found that even quite simple Transformer models display a variety of interesting behavior.
Understanding it seems like the bottom rungs of the ladder to unlocking deeper interpretability and control of large models.
[See some live examples!](https://tlab.dereklarson.info).
Note the double-descent in the extremely simple "Parity" experiment,
as it learns to categorize odds and evens via the embedding.
Also look at the multiple attention styles in "Addition29" when comparing across seeds.

One goal here is to highlight how powerful vision can be to generate clues to understanding a system.

## Development

After cloning the repo, simply run
- `npm i` to install dependencies
- `mkdir -p dist/data && cp sample_data/* dist/data` to prep the sample data
- `npm run dev` to launch the webpack dev server with HMR

