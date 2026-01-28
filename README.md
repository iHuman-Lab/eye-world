# Eye World

**Teaching machines to see where you see!**

Ever wondered what it would be like if your computer knew *exactly* where you were looking while gaming? Welcome to Eye World — where we train neural networks to predict your gaze while you munch ghosts in Ms. Pac-Man!

## What is this?

Eye World is a deep learning project that predicts **where humans look** in video game frames using eye-tracking data. We take raw gameplay footage + gaze coordinates and train models to understand visual attention patterns.

Think of it as teaching AI to develop a sixth sense for human eyeballs. Creepy? Maybe. Cool? Absolutely.


## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-username/eye-world.git
cd eye-world

# Install dependencies
pip install -r requirements.txt

# Configure your paths in configs/config.yaml

# Run the magic
python src/main.py
```

## Project Structure

```
eye-world/
├── src/
│   ├── main.py              # The main show
│   ├── dataset/             # Data wrangling
│   ├── models/              # Neural network architectures
│   ├── trainers/            # PyTorch Lightning training
│   └── data/                # Data processing utilities
├── configs/
│   └── config.yaml          # All the knobs and dials
├── data/
│   ├── raw/                 # Raw game footage
│   └── processed/           # Processed WebDatasets
└── tb_logs/                 # TensorBoard logs
```

## Training Metrics

Fire up TensorBoard to watch your models learn to see:

```bash
tensorboard --logdir tb_logs/
```

Watch beautiful heatmaps evolve as your network learns where humans look!

## Tech Stack

- **PyTorch Lightning** — Because training loops are boring
- **WebDataset** — For that sweet, sweet I/O performance
- **TensorBoard** — Pretty graphs go brrr

## Contributing

Found a bug? Have an idea? Want to add support for your favorite game?

1. Fork it
2. Branch it (`git checkout -b feature/awesome-thing`)
3. Commit it (`git commit -m 'Add awesome thing'`)
4. Push it (`git push origin feature/awesome-thing`)
5. PR it

## License

MIT. Do cool research with it. Make the world better. Be excellent to each other.

---

<p align="center">
  <i>Made with mass amounts of coffee and a concerning amount of eye-tracking data</i>
  <br>
  <b>iHuman Lab</b>
</p>
