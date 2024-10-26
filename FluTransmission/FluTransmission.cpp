#include <iostream>
#include <fstream>

const int gridHeight = 15;  // Grid height
const int gridWidth = 20;   // Grid width
double alpha = 0.1;          // Initial sick ratio
double beta = 0.3;           // Likelihood of transmission
int omega = 3;               // Days a person stays sick
int numDays = 10;            // Total simulation days

// Custom pseudo-random number generator
unsigned int customRand() {
    static unsigned int seed = 123456789;  // Seed value for random number generation
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;  // Return a pseudo-random number in the range [0, 32767]
}

// Function to initialize the grid with sick individuals based on initial ratio α
void initializeGrid(int** grid, int height, int width, double alpha) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int randVal = customRand() % 1000;
            grid[i][j] = (randVal < alpha * 1000) ? 1 : 0;  // Sick person or Healthy person
        }
    }
}

// Function to print the grid to a file (ASCII output)
void printGridToFile(int** grid, int height, int width, int day) {
    std::ofstream file("flu_simulation.txt", std::ios::app);  // Append to file
    file << "Day " << day << ":\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << grid[i][j] << " ";
        }
        file << "\n";
    }
    file.close();
}

// Function to update the grid based on transmission and recovery rules
void updateGrid(int** grid, int** newGrid, int height, int width, double beta, int omega, int** sickDays) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (grid[i][j] == 1) {  // If person is sick
                sickDays[i][j]++;
                if (sickDays[i][j] >= omega) {
                    newGrid[i][j] = 0;  // Recover
                    sickDays[i][j] = 0;
                }
                else {
                    newGrid[i][j] = 1;  // Stay sick
                }
            }
            else {
                int sickNeighbors = 0;
                if (i > 0 && grid[i - 1][j] == 1) sickNeighbors++;
                if (i < height - 1 && grid[i + 1][j] == 1) sickNeighbors++;
                if (j > 0 && grid[i][j - 1] == 1) sickNeighbors++;
                if (j < width - 1 && grid[i][j + 1] == 1) sickNeighbors++;

                int randVal = customRand() % 1000;
                newGrid[i][j] = (randVal < beta * sickNeighbors * 1000) ? 1 : 0;  // Get infected or Stay healthy
                if (newGrid[i][j] == 1) sickDays[i][j] = 1;  // Start counting sick days
            }
        }
    }
}

int main() {
    // Open the file in truncate mode initially to clear previous results
    std::ofstream file("flu_simulation.txt", std::ios::trunc);
    file.close();

    // Allocate grid and tracking arrays
    int** grid = new int* [gridHeight];
    int** newGrid = new int* [gridHeight];
    int** sickDays = new int* [gridHeight];
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = new int[gridWidth];
        newGrid[i] = new int[gridWidth];
        sickDays[i] = new int[gridWidth] {0};  // Initialize sick days to 0
    }

    // Initialize the grid
    initializeGrid(grid, gridHeight, gridWidth, alpha);

    // Simulation loop
    for (int day = 0; day < numDays; ++day) {
        // Update the grid based on transmission/recovery rules
        updateGrid(grid, newGrid, gridHeight, gridWidth, beta, omega, sickDays);

        // Print grid to file
        printGridToFile(newGrid, gridHeight, gridWidth, day);

        // Swap grid and newGrid pointers for the next iteration
        int** temp = grid;
        grid = newGrid;
        newGrid = temp;
    }

    // Free dynamically allocated memory
    for (int i = 0; i < gridHeight; ++i) {
        delete[] grid[i];
        delete[] newGrid[i];
        delete[] sickDays[i];
    }
    delete[] grid;
    delete[] newGrid;
    delete[] sickDays;

    return 0;
}
