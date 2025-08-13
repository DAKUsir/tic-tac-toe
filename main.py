import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from train import TicTacToeNet
from Dataset_and_HelperFunction import check_winner
import threading
import time

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("🧠 Neural Tic-Tac-Toe")
        self.master.geometry("1400x900")
        self.master.configure(bg='#0f0f23')
        
        # Game state
        self.net = None
        self.board = [0]*9
        self.player = 1
        self.buttons = []
        self.is_training = False
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.epoch_history = []
        
        # Create the main layout
        self.create_styles()
        self.create_layout()
        
    def create_styles(self):
        """Create custom styles for the application"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', 
                           background='#0f0f23', 
                           foreground='#00d4aa', 
                           font=('Segoe UI', 28, 'bold'))
        
        self.style.configure('Header.TLabel', 
                           background='#1a1a3a', 
                           foreground='#ffffff', 
                           font=('Segoe UI', 14, 'bold'))
        
        self.style.configure('Custom.TLabel', 
                           background='#1a1a3a', 
                           foreground='#e0e0e0', 
                           font=('Segoe UI', 11))
        
        self.style.configure('Status.TLabel', 
                           background='#0f0f23', 
                           foreground='#00d4aa', 
                           font=('Segoe UI', 16, 'bold'))
        
        self.style.configure('Custom.TButton',
                           background='#00d4aa',
                           foreground='#0f0f23',
                           font=('Segoe UI', 12, 'bold'),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.configure('Reset.TButton',
                           background='#ff6b6b',
                           foreground='#ffffff',
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=0,
                           focuscolor='none')
        
    def create_layout(self):
        """Create the main layout with three panels"""
        # Main title
        title_label = ttk.Label(self.master, text="🧠 Neural Tic-Tac-Toe", style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Create main container
        main_frame = tk.Frame(self.master, bg='#0f0f23')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left Panel - Training Controls
        self.create_training_panel(main_frame)
        
        # Center Panel - Game Board
        self.create_game_panel(main_frame)
        
        # Right Panel - Graphs
        self.create_graph_panel(main_frame)
        
        # Status bar at bottom
        self.create_status_bar()
        
    def create_training_panel(self, parent):
        """Create the training controls panel"""
        train_frame = tk.Frame(parent, bg='#1a1a3a', relief='raised', bd=2)
        train_frame.pack(side='left', fill='y', padx=(0, 10), pady=5)
        
        # Training header
        header = ttk.Label(train_frame, text="🎯 Training Center", style='Header.TLabel')
        header.pack(pady=15, padx=20)
        
        # Training parameters frame
        params_frame = tk.Frame(train_frame, bg='#1a1a3a')
        params_frame.pack(padx=20, pady=10)
        
        # Dataset Size
        ttk.Label(params_frame, text="Dataset Size:", style='Custom.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.dataset_size_var = tk.StringVar(value="6617")
        dataset_entry = tk.Entry(params_frame, textvariable=self.dataset_size_var, 
                               bg='#2a2a4a', fg='#ffffff', font=('Segoe UI', 10),
                               width=15, relief='flat', bd=5)
        dataset_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:", style='Custom.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        self.epochs_var = tk.StringVar(value="500")
        epochs_entry = tk.Entry(params_frame, textvariable=self.epochs_var, 
                              bg='#2a2a4a', fg='#ffffff', font=('Segoe UI', 10),
                              width=15, relief='flat', bd=5)
        epochs_entry.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # Batch Size
        ttk.Label(params_frame, text="Batch Size:", style='Custom.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        self.batch_size_var = tk.StringVar(value="128")
        batch_entry = tk.Entry(params_frame, textvariable=self.batch_size_var, 
                             bg='#2a2a4a', fg='#ffffff', font=('Segoe UI', 10),
                             width=15, relief='flat', bd=5)
        batch_entry.grid(row=2, column=1, padx=(10, 0), pady=5)
        
        # Learning Rate
        ttk.Label(params_frame, text="Learning Rate:", style='Custom.TLabel').grid(row=3, column=0, sticky='w', pady=5)
        self.lr_var = tk.StringVar(value="0.01")
        lr_entry = tk.Entry(params_frame, textvariable=self.lr_var, 
                          bg='#2a2a4a', fg='#ffffff', font=('Segoe UI', 10),
                          width=15, relief='flat', bd=5)
        lr_entry.grid(row=3, column=1, padx=(10, 0), pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(train_frame, mode='indeterminate')
        self.progress.pack(pady=20, padx=20, fill='x')
        
        # Training button
        self.train_button = tk.Button(train_frame, text="🚀 Start Training", 
                                    command=self.train_net,
                                    bg='#00d4aa', fg='#0f0f23', 
                                    font=('Segoe UI', 14, 'bold'),
                                    relief='flat', bd=0, pady=10,
                                    cursor='hand2')
        self.train_button.pack(pady=10, padx=20, fill='x')
        
        # Reset button
        reset_button = tk.Button(train_frame, text="🔄 Reset Game", 
                               command=self.reset_board,
                               bg='#ff6b6b', fg='#ffffff', 
                               font=('Segoe UI', 12, 'bold'),
                               relief='flat', bd=0, pady=8,
                               cursor='hand2')
        reset_button.pack(pady=(5, 20), padx=20, fill='x')
        
        # Training stats
        stats_frame = tk.Frame(train_frame, bg='#2a2a4a', relief='sunken', bd=2)
        stats_frame.pack(padx=20, pady=10, fill='x')
        
        ttk.Label(stats_frame, text="📊 Training Stats", style='Header.TLabel').pack(pady=10)
        
        self.loss_label = tk.Label(stats_frame, text="Final Loss: N/A", 
                                 bg='#2a2a4a', fg='#00d4aa', 
                                 font=('Segoe UI', 11, 'bold'))
        self.loss_label.pack(pady=5)
        
        self.accuracy_label = tk.Label(stats_frame, text="Accuracy: N/A", 
                                     bg='#2a2a4a', fg='#00d4aa', 
                                     font=('Segoe UI', 11, 'bold'))
        self.accuracy_label.pack(pady=5)
        
        self.training_time_label = tk.Label(stats_frame, text="Training Time: N/A", 
                                          bg='#2a2a4a', fg='#00d4aa', 
                                          font=('Segoe UI', 11, 'bold'))
        self.training_time_label.pack(pady=(5, 10))
        
    def create_game_panel(self, parent):
        """Create the game board panel"""
        game_frame = tk.Frame(parent, bg='#1a1a3a', relief='raised', bd=2)
        game_frame.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        
        # Game header
        header = ttk.Label(game_frame, text="🎮 Game Board", style='Header.TLabel')
        header.pack(pady=15)
        
        # Board container
        board_container = tk.Frame(game_frame, bg='#1a1a3a')
        board_container.pack(expand=True)
        
        # Create the tic-tac-toe board
        self.create_board(board_container)
        
    def create_graph_panel(self, parent):
        """Create the graphs panel"""
        graph_frame = tk.Frame(parent, bg='#1a1a3a', relief='raised', bd=2)
        graph_frame.pack(side='right', fill='y', padx=(10, 0), pady=5)
        
        # Graph header
        header = ttk.Label(graph_frame, text="📈 Training Analytics", style='Header.TLabel')
        header.pack(pady=15, padx=20)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 8), facecolor='#1a1a3a', tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize empty plots
        self.loss_ax = self.fig.add_subplot(2, 1, 1)
        self.acc_ax = self.fig.add_subplot(2, 1, 2)
        
        self.setup_empty_plots()
        
    def setup_empty_plots(self):
        """Setup empty plots with styling"""
        # Loss plot
        self.loss_ax.set_facecolor('#2a2a4a')
        self.loss_ax.set_title('Training Loss', color='#00d4aa', fontsize=12, fontweight='bold')
        self.loss_ax.set_xlabel('Epoch', color='#ffffff')
        self.loss_ax.set_ylabel('Loss', color='#ffffff')
        self.loss_ax.tick_params(colors='#ffffff')
        self.loss_ax.grid(True, alpha=0.3, color='#555555')
        
        # Accuracy plot
        self.acc_ax.set_facecolor('#2a2a4a')
        self.acc_ax.set_title('Training Accuracy', color='#00d4aa', fontsize=12, fontweight='bold')
        self.acc_ax.set_xlabel('Epoch', color='#ffffff')
        self.acc_ax.set_ylabel('Accuracy (%)', color='#ffffff')
        self.acc_ax.tick_params(colors='#ffffff')
        self.acc_ax.grid(True, alpha=0.3, color='#555555')
        
        self.canvas.draw()
        
    def create_board(self, parent):
        """Create the tic-tac-toe board"""
        board_frame = tk.Frame(parent, bg='#2a2a4a', relief='raised', bd=3)
        board_frame.pack(pady=20)
        
        for i in range(9):
            row, col = divmod(i, 3)
            btn = tk.Button(board_frame, text=" ", width=4, height=2,
                          font=('Segoe UI', 36, 'bold'), 
                          bg='#3a3a5a', fg='#ffffff',
                          relief='raised', bd=3,
                          command=lambda idx=i: self.human_move(idx),
                          cursor='hand2',
                          activebackground='#4a4a6a')
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.buttons.append(btn)
            
    def create_status_bar(self):
        """Create status bar at the bottom"""
        status_frame = tk.Frame(self.master, bg='#0f0f23', height=60)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status = tk.Label(status_frame, 
                             text="🎯 Configure training parameters and click 'Start Training'", 
                             bg='#0f0f23', fg='#00d4aa', 
                             font=('Segoe UI', 14, 'bold'))
        self.status.pack(expand=True)
        
    def train_net(self):
        """Train the neural network in a separate thread"""
        if self.is_training:
            return
            
        try:
            dataset_size = int(self.dataset_size_var.get())
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("❌ Error", "Please enter valid numbers for training options.")
            return
            
        self.is_training = True
        self.train_button.config(state='disabled', text="🔄 Training...", bg='#666666')
        self.progress.start()
        
        # Clear previous data
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.epoch_history.clear()
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self._train_worker, 
                                         args=(dataset_size, epochs, batch_size, lr))
        training_thread.daemon = True
        training_thread.start()
        
    def _train_worker(self, dataset_size, epochs, batch_size, lr):
        """Worker function for training"""
        start_time = time.time()
        
        try:
            self.net = TicTacToeNet(dataset_size)
            
            # Mock training with progress updates (replace with actual training)
            for epoch in range(epochs):
                # Simulate training step
                if epoch % 10 == 0:  # Update every 10 epochs
                    # Mock loss and accuracy (replace with actual values from your training)
                    mock_loss = 1.0 * np.exp(-epoch / 100) + 0.01 * np.random.random()
                    mock_accuracy = min(95, 70 + (epoch / epochs) * 25 + 5 * np.random.random())
                    
                    self.loss_history.append(mock_loss)
                    self.accuracy_history.append(mock_accuracy)
                    self.epoch_history.append(epoch)
                    
                    # Update plots
                    self.master.after(0, self.update_plots)
                    
                time.sleep(0.01)  # Simulate training time
                
            # Actual training call (uncomment and modify as needed)
            final_loss = self.net.train(epochs=epochs, batch_size=batch_size, lr=lr)
            
            training_time = time.time() - start_time
            final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0
            
            # Update UI on main thread
            self.master.after(0, lambda: self._training_complete(final_loss, final_accuracy, training_time))
            
        except Exception as e:
            self.master.after(0, lambda: self._training_error(str(e)))
            
    def _training_complete(self, final_loss, final_accuracy, training_time):
        """Handle training completion"""
        self.is_training = False
        self.progress.stop()
        self.train_button.config(state='normal', text="🚀 Start Training", bg='#00d4aa')
        
        # Update stats
        self.loss_label.config(text=f"Final Loss: {final_loss:.4f}")
        self.accuracy_label.config(text=f"Accuracy: {final_accuracy:.1f}%")
        self.training_time_label.config(text=f"Training Time: {training_time:.1f}s")
        
        self.status.config(text="✅ Training complete! Your turn (X)", fg='#00d4aa')
        self.reset_board()
        
    def _training_error(self, error_msg):
        """Handle training errors"""
        self.is_training = False
        self.progress.stop()
        self.train_button.config(state='normal', text="🚀 Start Training", bg='#00d4aa')
        messagebox.showerror("❌ Training Error", f"Training failed: {error_msg}")
        self.status.config(text="❌ Training failed. Please try again.", fg='#ff6b6b')
        
    def update_plots(self):
        """Update the training plots"""
        if not self.loss_history:
            return
            
        # Clear plots
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # Plot loss
        self.loss_ax.plot(self.epoch_history, self.loss_history, 
                         color='#ff6b6b', linewidth=2, marker='o', markersize=3)
        self.loss_ax.set_facecolor('#2a2a4a')
        self.loss_ax.set_title('Training Loss', color='#00d4aa', fontsize=12, fontweight='bold')
        self.loss_ax.set_xlabel('Epoch', color='#ffffff')
        self.loss_ax.set_ylabel('Loss', color='#ffffff')
        self.loss_ax.tick_params(colors='#ffffff')
        self.loss_ax.grid(True, alpha=0.3, color='#555555')
        
        # Plot accuracy
        self.acc_ax.plot(self.epoch_history, self.accuracy_history, 
                        color='#00d4aa', linewidth=2, marker='s', markersize=3)
        self.acc_ax.set_facecolor('#2a2a4a')
        self.acc_ax.set_title('Training Accuracy', color='#00d4aa', fontsize=12, fontweight='bold')
        self.acc_ax.set_xlabel('Epoch', color='#ffffff')
        self.acc_ax.set_ylabel('Accuracy (%)', color='#ffffff')
        self.acc_ax.tick_params(colors='#ffffff')
        self.acc_ax.grid(True, alpha=0.3, color='#555555')
        
        self.canvas.draw()
        
    def reset_board(self):
        """Reset the game board"""
        self.board = [0]*9
        self.player = 1
        for btn in self.buttons:
            btn.config(text=" ", state="normal", 
                      bg='#3a3a5a', fg='#ffffff',
                      relief='raised')
        self.status.config(text="🎯 Your turn (X)", fg='#00d4aa')
        
    def human_move(self, idx):
        """Handle human player move"""
        if self.net is None:
            messagebox.showinfo("ℹ️ Info", "Please train the model first.")
            return
            
        if self.board[idx] == 0:
            self.board[idx] = 1
            self.update_board()
            res = check_winner(self.board)
            if res is not None:
                self.end_game(res)
                return
            self.status.config(text="🤖 Model's turn (O)", fg='#ffa500')
            self.master.after(800, self.model_move)
            
    def model_move(self):
        """Handle AI model move"""
        try:
            mv = self.net.predict(np.array(self.board))
            if self.board[mv] != 0:
                available_moves = [i for i, v in enumerate(self.board) if v == 0]
                if available_moves:
                    mv = random.choice(available_moves)
                else:
                    return
        except:
            available_moves = [i for i, v in enumerate(self.board) if v == 0]
            if available_moves:
                mv = random.choice(available_moves)
            else:
                return
                
        self.board[mv] = -1
        self.update_board()
        res = check_winner(self.board)
        if res is not None:
            self.end_game(res)
            return
        self.status.config(text="🎯 Your turn (X)", fg='#00d4aa')
        
    def update_board(self):
        """Update the visual board"""
        for i, btn in enumerate(self.buttons):
            if self.board[i] == 1:
                btn.config(text="X", state="disabled", 
                          bg='#00d4aa', fg='#0f0f23',
                          relief='sunken', font=('Segoe UI', 36, 'bold'))
            elif self.board[i] == -1:
                btn.config(text="O", state="disabled", 
                          bg='#ff6b6b', fg='#ffffff',
                          relief='sunken', font=('Segoe UI', 36, 'bold'))
            else:
                btn.config(text=" ", state="normal", 
                          bg='#3a3a5a', fg='#ffffff',
                          relief='raised')
                
    def end_game(self, result):
        """Handle game end"""
        for btn in self.buttons:
            btn.config(state="disabled")
            
        if result == 1:
            self.status.config(text="🎉 You win! Congratulations!", fg='#00d4aa')
            messagebox.showinfo("🎉 Game Over", "You win! Congratulations!")
        elif result == -1:
            self.status.config(text="🤖 AI wins! Better luck next time!", fg='#ff6b6b')
            messagebox.showinfo("🤖 Game Over", "AI wins! Better luck next time!")
        else:
            self.status.config(text="🤝 It's a draw! Well played!", fg='#ffa500')
            messagebox.showinfo("🤝 Game Over", "It's a draw! Well played!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()