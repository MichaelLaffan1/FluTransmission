import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_flu_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        current_day = []
        
        for line in file:
            if line.startswith("Day"):
                if current_day:
                    data.append(np.array(current_day))
                    current_day = []
            else:
                current_day.append(list(map(int, line.split())))
        
        # Append the last day if not added
        if current_day:
            data.append(np.array(current_day))
    
    return data

def animate_flu_data(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create an initial plot using the first day's data
    im = ax.imshow(data[0], cmap='coolwarm', interpolation='nearest')
    title = ax.text(0.5, 1.05, "Day 0", size=16, ha="center", transform=ax.transAxes)
    ax.set_aspect('equal')

    def update(day):
        im.set_data(data[day])
        title.set_text(f"Day {day}")
        return [im, title]

    # Create the animation with a 1.5-second delay between frames
    anim = FuncAnimation(fig, update, frames=len(data), interval=1500, blit=True)

    plt.show()

def plot_flu_data(data):
    days = len(data)
    cols = int(np.ceil(np.sqrt(days)))
    rows = int(np.ceil(days / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Flatten the axs array if there are multiple rows
    axs = axs.flatten() if rows > 1 else [axs]

    for day_idx, (ax, day_data) in enumerate(zip(axs, data)):
        ax.imshow(day_data, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Day {day_idx}")
        ax.set_aspect('equal') # Make each plot square

    # Hide unused subplots in case of an uneven grid
    for ax in axs[len(data):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "FluTransmission/flu_simulation.txt"
    flu_data = read_flu_data(file_path)
    
    plot_flu_data(flu_data)
    animate_flu_data(flu_data)
