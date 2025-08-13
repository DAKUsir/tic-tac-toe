
---

# 🧠 Tic-Tac-Toe Neural Agent (Pure NumPy + Tkinter)

A from-scratch neural network that learns perfect Tic-Tac-Toe play via minimax-generated training data, implemented entirely with **NumPy**.
Includes a **Tkinter-based GUI** for an interactive human vs AI gameplay experience.

---

## 🚀 Features

* **Pure NumPy neural network** (no external ML frameworks)
* **Forward & backward propagation** implemented manually
* **Trained on perfect-move dataset** generated using minimax algorithm
* **Tkinter GUI** with clickable grid for real-time play
* AI plays optimally — hard to beat, always blocks losing moves

---

## 🛠 Project Structure

```
├── Dataset_and_HelperFunction.py   # Dataset generation & helper functions
├── Numpy_Neural_net.py              # Neural network architecture & training logic
├── main.py                          # Tkinter GUI game loop
├── train.py                         # Model training script
├── requirements.txt                 # Dependencies
└── __pycache__/                     # Compiled cache files
```

---

## 🎯 How It Works

1. **Board Representation:**

   * 9 inputs: `1` for player, `-1` for opponent, `0` for empty.
2. **Neural Network Output:**

   * 9 values (one per cell) → AI chooses highest-scoring valid move.
3. **Training:**

   * Dataset of optimal moves generated with minimax algorithm.
   * Trained using gradient descent and manual backpropagation.
4. **Gameplay:**

   * User plays as `X`, AI as `O`.
   * GUI updates in real time and announces results.

---

## 📦 Installation & Usage

```bash
# Clone repository
git clone https://github.com/<your-username>/tic-tac-toe.git
cd tic-tac-toe

# Install dependencies
pip install -r requirements.txt

# Run the game
python main.py
```

---


## 📌 Roadmap

* [ ] Add difficulty levels
* [ ] Add sound effects
* [ ] Keep score between matches
* [ ] Improve AI through reinforcement learning self-play

---

## 📜 License

MIT License — feel free to use, modify, and share.

---

If you want, I can also make a **GIF of your Tkinter gameplay** so it looks more engaging on GitHub. That usually helps people instantly see what the project does.
Do you want me to prepare that?
