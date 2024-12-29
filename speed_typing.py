import tkinter as tk
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque


class TypingEnvironment:
    def __init__(self, words, time_limit):
        self.words = words
        self.time_limit = time_limit
        self.reset()

    def reset(self):
        self.time_left = self.time_limit
        self.score = 0
        self.current_word = random.choice(self.words)
        return self.get_state()

    def get_state(self):
        state = np.zeros(len(self.words))
        if self.current_word is not None:
            try:
                state[self.words.index(self.current_word)] = 1  # One-hot encode the current word
            except ValueError:
                pass
        return state

    def step(self, action):
        reward = 0
        done = False
        if self.current_word is not None and self.words[action] == self.current_word:
            reward = 1
            self.score += 1
        else:
            reward = -1

        self.time_left -= 1
        if self.time_left <= 0:
            done = True
            self.current_word = None  # Game over, no current word
        else:
            self.current_word = random.choice(self.words)

        return (self.get_state() if not done else np.zeros(len(self.words))), reward, done


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))  # Smaller network
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done and next_state is not None:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TypingGameGUI:
    def __init__(self, root, env, agent):
        self.root = root
        self.env = env
        self.agent = agent
        self.root.title("Speed Typing Game")
        self.root.geometry("400x300")
        self.root.resizable(False, False)

        self.time_limit = env.time_limit
        self.start_time = None
        self.score = 0
        self.current_word = ""
        self.running = False

        # Set the background color
        self.root.configure(bg="black")

        # Layout
        self.label_timer = tk.Label(root, text=f"Time: {self.time_limit}s", font=("Helvetica", 16), fg="white", bg="black")
        self.label_timer.pack()

        self.label_word = tk.Label(root, text="Press Start to Play", font=("Helvetica", 20, "bold"), fg="white", bg="black")
        self.label_word.pack(pady=20)

        self.entry = tk.Entry(root, font=("Helvetica", 16), justify="center", bg="black", fg="white", insertbackground="white")
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", self.check_word)
        self.entry.configure(state="disabled")

        self.button_start = tk.Button(root, text="Start", font=("Helvetica", 14), command=self.start_game, fg="white", bg="black",
                                      activeforeground="white", activebackground="gray")
        self.button_start.pack(pady=10)

        self.label_score = tk.Label(root, text=f"Score: {self.score}", font=("Helvetica", 16), fg="white", bg="black")
        self.label_score.pack()

    def start_game(self):
        self.env.reset()
        self.score = 0
        self.running = True
        self.update_ui()
        self.entry.configure(state="normal")
        self.entry.focus()

    def check_word(self, event):
        typed_word = self.entry.get()
        self.entry.delete(0, tk.END)

        if typed_word in self.env.words:
            action = self.env.words.index(typed_word)
        else:
            action = -1

        if action != -1:
            state, reward, done = self.env.step(action)
            self.agent.remember(state, action, reward, state if not done else None, done)
            self.agent.replay(32)

            if done:
                self.running = False
                self.label_word.config(text="Game Over!")
                self.entry.configure(state="disabled")
                return

        self.update_ui()

    def update_ui(self):
        self.label_timer.config(text=f"Time: {self.env.time_left}s")
        self.label_score.config(text=f"Score: {self.env.score}")
        self.label_word.config(text=self.env.current_word if self.running else "Game Over!")


if __name__ == "__main__":
    words = ["python", "developer", "speed", "game", "typing", "keyboard", "accuracy", "program", "challenge", "function", "variable", "loop"]
    time_limit = 30

    env = TypingEnvironment(words, time_limit)
    state_size = len(words)
    action_size = len(words)
    agent = DQNAgent(state_size, action_size)

    root = tk.Tk()
    gui = TypingGameGUI(root, env, agent)
    root.mainloop()
