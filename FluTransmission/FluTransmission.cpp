#include <iostream>
#include <fstream>
#include <omp.h>

const int gridHeight = 15;  // Grid height
const int gridWidth = 20;   // Grid width
double alpha = 0.1;          // Initial sick ratio
double beta = 0.3;           // Likelihood of transmission
int omega = 3;               // Days a person stays sick
int numDays = 10;            // Total simulation days

// Class to represent a person
class Person {
public:
    bool was_infected;
    int sick_days;

    Person() : was_infected(false), sick_days(0) {}
};

// Custom pseudo-random number generator
unsigned int customRand() {
    static unsigned int seed = 123456789;  // Seed value for random number generation
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;  // Return a pseudo-random number in the range [0, 32767]
}

// Function to initialize the grid with sick individuals based on initial ratio α
void initializeGrid(Person grid[gridHeight][gridWidth], double alpha) {
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            int randVal = customRand() % 1000;
            if (randVal < alpha * 1000) {
                grid[i][j].was_infected = true;  // Initially sick person
                grid[i][j].sick_days = 1;        // Initialize sick days
            }
        }
    }
}

// Function to print the grid to a file (ASCII output)
void printGridToFile(Person grid[gridHeight][gridWidth], int day) {
    std::ofstream file("flu_simulation.txt", std::ios::app);  // Append to file
    file << "Day " << day << ":\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << (grid[i][j].sick_days > 0 ? 1 : 0) << " ";
        }
        file << "\n";
    }
    file.close();
}

// Function to update the grid based on transmission and recovery rules
void updateGrid(Person grid[gridHeight][gridWidth], Person newGrid[gridHeight][gridWidth]) {
    #pragma omp parallel for collapse(2) // collapse(2) allows OpenMP to consider the nested loops as one loop for better thread distribution
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            newGrid[i][j] = grid[i][j]; // Copy current state

            if (grid[i][j].sick_days > 0) {  // If person is sick
                newGrid[i][j].sick_days++;  // Increment sick days
                if (newGrid[i][j].sick_days >= omega) {  // Check if they recover
                    newGrid[i][j].sick_days = 0;  // Person recovers (not sick anymore)
                    // 'was_infected' remains true, preventing re-infection
                }
            }
            else if (!grid[i][j].was_infected) {  // Only never-infected individuals can get infected
                int sickNeighbors = 0;
                if (i > 0 && grid[i - 1][j].sick_days > 0) sickNeighbors++;
                if (i < gridHeight - 1 && grid[i + 1][j].sick_days > 0) sickNeighbors++;
                if (j > 0 && grid[i][j - 1].sick_days > 0) sickNeighbors++;
                if (j < gridWidth - 1 && grid[i][j + 1].sick_days > 0) sickNeighbors++;

                int randVal = customRand() % 1000;
                if (randVal < beta * sickNeighbors * 1000) {
                    newGrid[i][j].was_infected = true;  // Get infected
                    newGrid[i][j].sick_days = 1;        // Start counting sick days
                }
            }
        }
    }
}

int main() {
    // Open the file in truncate mode initially to clear previous results
    std::ofstream file("flu_simulation.txt", std::ios::trunc);
    file.close();

    /* Allocate grid and tracking arrays. Having grid and newGrid allows for:
        1. Consistent calculations by ensuring each day’s updates are based on a static snapshot of the previous day.
        2. Correct modeling of simultaneous infection / recovery without interference from updates that have not yet been fully applied.
    */
    Person grid[gridHeight][gridWidth];
    Person newGrid[gridHeight][gridWidth];

    initializeGrid(grid, alpha);

    // Simulation loop
    for (int day = 0; day < numDays; ++day) {
        // Parallel sections: one for updating the grid, another for writing to file
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Print the last day's grid to file while the next day's grid is being computed
                if (day > 0) {
                    printGridToFile(newGrid, day - 1);
                }
            }
            #pragma omp section
            {
                // Update the grid based on transmission/recovery rules
                updateGrid(grid, newGrid);
            }
        }

        // Swap grid and newGrid pointers for the next iteration
        for (int i = 0; i < gridHeight; ++i) {
            for (int j = 0; j < gridWidth; ++j) {
                grid[i][j] = newGrid[i][j]; // Copy newGrid back to grid
            }
        }
    }

    // Print the last day's grid to file
    printGridToFile(newGrid, numDays - 1);

    return 0;
}
