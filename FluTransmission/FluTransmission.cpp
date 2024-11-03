#include <iostream>
#include <fstream>
#include <omp.h>

// Default values for simulation settings
int gridHeight = 5;  // Grid height
int gridWidth = 5;   // Grid width
double alpha = 0.1;   // Initial sick ratio
double beta = 0.3;    // Likelihood of transmission
int omega = 2;        // Days a person stays sick
int numDays = 5;     // Total simulation days
int numThreads = omp_get_max_threads();   // Max threads for OpenMP

// Class to represent a person
class Person {
public:
    bool was_infected;
    int sick_days;

    Person() : was_infected(false), sick_days(0) {}
};

// Function to read simulation settings from a configuration file
void readSettingsFromFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open settings file.\n";
        return;
    }

    // Each line in the file is expected to correspond to a specific setting
    char label[50];
    if (file >> label >> gridHeight &&
        file >> label >> gridWidth &&
        file >> label >> alpha &&
        file >> label >> beta &&
        file >> label >> omega &&
        file >> label >> numDays &&
        file >> label >> numThreads) {

        if (numThreads == 0) {
            numThreads = omp_get_max_threads();
        }
    }
    else {
        std::cerr << "Error reading settings file.\n";
    }

    file.close();
}

// Custom pseudo-random number generator
unsigned int customRand() {
    static unsigned int seed = 123456789;  // Seed value for random number generation
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;  // Return a pseudo-random number in the range [0, 32767]
}

// Function to initialize the grid with sick individuals based on initial ratio α
void initializeGrid(Person** grid, double alpha) {
    int totalPeople = gridHeight * gridWidth;
    int infectedCount = static_cast<int>(alpha * totalPeople);
    std::cout << "infected count " << infectedCount << std::endl;

    // Ensure that we do not place more infected individuals than the grid size
    if (infectedCount > totalPeople) {
        infectedCount = totalPeople;
    }

    int placedInfected = 0;

    // Randomly place infected individuals in the grid
    while (placedInfected < infectedCount) {
        int i = customRand() % gridHeight;
        int j = customRand() % gridWidth;

        // Only place an infected person if the spot is not already infected
        if (!grid[i][j].was_infected) {
            grid[i][j].was_infected = true;
            grid[i][j].sick_days = 1;
            placedInfected++;
        }
    }
}

// Function to print the grid to a file (ASCII output)
void printGridToFile(Person** grid, int day) {
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
void updateGrid(Person** grid, Person** newGrid) {
    #pragma omp parallel for collapse(2) // collapse(2) allows OpenMP to consider the nested loops as one loop for better thread distribution
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            newGrid[i][j] = grid[i][j]; // Copy current state

            if (grid[i][j].sick_days > 0) {  // If person is sick
                newGrid[i][j].sick_days++;  // Increment sick days
                if (newGrid[i][j].sick_days >= omega) {  // Check if they recover
                    newGrid[i][j].sick_days = 0;  // Person recovers (not sick anymore)
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
    // Read settings from configuration file
    readSettingsFromFile("settings.txt");

    // Set the number of threads for OpenMP
    omp_set_num_threads(numThreads);

    // Open the file in truncate mode initially to clear previous results
    std::ofstream file("flu_simulation.txt", std::ios::trunc);
    file.close();

    /* Allocate grid and tracking arrays. Having grid and newGrid allows for:
      1. Consistent calculations by ensuring each day’s updates are based on a static snapshot of the previous day.
      2. Correct modeling of simultaneous infection / recovery without interference from updates that have not yet been fully applied.
    */
    Person** grid = new Person * [gridHeight];
    Person** newGrid = new Person * [gridHeight];
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = new Person[gridWidth];
        newGrid[i] = new Person[gridWidth];
    }

    initializeGrid(grid, alpha);
    printGridToFile(grid, 0);

    // Simulation loop
    for (int day = 0; day < numDays; ++day) {
        // Parallel sections: one for updating the grid, another for writing to file
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Print the last day's grid to file while the next day's grid is being computed
                if (day > 0) {
                    printGridToFile(newGrid, day);
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

    // Free dynamically allocated memory
    for (int i = 0; i < gridHeight; ++i) {
        delete[] grid[i];
        delete[] newGrid[i];
    }
    delete[] grid;
    delete[] newGrid;

    return 0;
}
