import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_flu_data(file_path):
    """Read flu simulation data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = []
            current_day = []
            
            for line in file:
                if line.startswith("Day"):
                    if current_day:  # Append previous day data if exists
                        data.append(np.array(current_day))
                        current_day = []
                else:
                    current_day.append(list(map(int, line.split())))
            
            # Append the last day if not added
            if current_day:
                data.append(np.array(current_day))

        # Debug: Print the number of days and their shapes
        print(f"Read {len(data)} days of data from {file_path}.")
        for idx, day in enumerate(data):
            print(f"Day {idx}: Shape {day.shape}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    
    return data

def animate_data(days):
    """Animate the flu simulation data over days."""
    if not days:
        print("No data available for animation.")
        return
    
    fig, ax = plt.subplots()
    im = ax.imshow(days[0], cmap='coolwarm', vmin=0, vmax=1)
    
    # Add text annotation for the day counter
    day_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

    def update(frame):
        im.set_data(days[frame])
        day_text.set_text(f'Day {frame}')  # Update day label for each frame
        return im, day_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(days), blit=True, interval=1000
    )
    plt.show()

def plot_data(data):
    """Plot the flu simulation data for each day."""
    if not data:
        print("No data available for plotting.")
        return

    days = len(data)
    cols = int(np.ceil(np.sqrt(days)))
    rows = int(np.ceil(days / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Flatten the axs array if there are multiple rows
    axs = axs.flatten() if rows > 1 else [axs]

    for day_idx, (ax, day_data) in enumerate(zip(axs, data)):
        ax.imshow(day_data, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Day {day_idx}")
        ax.set_aspect('equal')  # Make each plot square

    # Hide unused subplots in case of an uneven grid
    for ax in axs[len(data):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    flu_data = read_flu_data("FluTransmission/flu_simulation.txt")
    plot_data(flu_data)
    animate_data(flu_data)

    debug_data = read_flu_data("FluTransmission/flu_debug.txt")
    plot_data(debug_data)
    animate_data(debug_data)
