#include <iostream>
#include <fstream>
#include <omp.h>

// Default values for simulation settings
int gridHeight = 5;
int gridWidth = 5;
double alpha = 0.1; // Initial sick ratio
double beta = 0.3; // Likelihood of transmission
int omega = 2; // Days a person stays sick
int numDays = 5; // Total simulation days
int numThreads; // Threads that OpenMP should use

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

// Thread-safe custom random generator using thread-local seed
unsigned int customRand(unsigned int& seed) {
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;
}

// Initialize the grid with sick individuals based on alpha, using thread-local random seeds
void initializeGrid(Person** grid) {
    int totalPeople = gridHeight * gridWidth;
    int infectedCount = static_cast<int>(alpha * totalPeople);

    #pragma omp parallel for schedule(guided)
    for (int count = 0; count < infectedCount; ++count) {
        unsigned int seed = 123456789 + omp_get_thread_num(); // Unique seed per thread
        int i, j;
        do {
            i = customRand(seed) % gridHeight;
            j = customRand(seed) % gridWidth;
        } while (grid[i][j].was_infected);
        grid[i][j].was_infected = true;
        grid[i][j].sick_days = 1;
    }
}

// Function to print the main grid to a file
void printGridToFile(Person** grid, int day) {
    std::ofstream file("flu_simulation.txt", std::ios::app); // Append to file
    file << "Day " << day << ":\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << (grid[i][j].sick_days > 0 ? 1 : 0) << " "; // sick status
        }
        file << "\n";
    }
    file.close();
}

// Function to print the thread grid to a separate file
void printThreadGridToFile(int** threadGrid, int day) {
    std::ofstream file("thread_grid.txt", std::ios::app);  // Append to file
    file << "Day " << day << ":\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << threadGrid[i][j] << " ";  // Write each cell's thread number
        }
        file << "\n";
    }
    file.close();
}

// Update grid based on transmission and recovery rules
void updateGrid(Person** grid, Person** newGrid, int** threadGrid) {
    #pragma omp parallel for collapse(2) schedule(static) // Use static scheduling to balance work across threads
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            newGrid[i][j] = grid[i][j]; // Copy current state

            int thread_id = omp_get_thread_num(); // Get thread ID
            threadGrid[i][j] = thread_id;         // Record which thread is working on this cell
            if (thread_id != 0)
                std::cout << "A different thread did something! " << thread_id << std::endl;

            if (grid[i][j].sick_days > 0) {  // If person is sick
                newGrid[i][j].sick_days++;  // Increment sick days
                if (newGrid[i][j].sick_days >= omega) { // Check if they recovered
                    newGrid[i][j].sick_days = 0;  // Person recovers
                }
            }
            else if (!grid[i][j].was_infected) {  // Only never-infected individuals can get infected
                int sickNeighbors = 0;
                if (i > 0 && grid[i - 1][j].sick_days > 0) sickNeighbors++;
                if (i < gridHeight - 1 && grid[i + 1][j].sick_days > 0) sickNeighbors++;
                if (j > 0 && grid[i][j - 1].sick_days > 0) sickNeighbors++;
                if (j < gridWidth - 1 && grid[i][j + 1].sick_days > 0) sickNeighbors++;

                unsigned int thread_seed = 123456789 + thread_id; // thread-specific seed
                if (customRand(thread_seed) % 1000 < beta * sickNeighbors * 1000) {
                    newGrid[i][j].was_infected = true;
                    newGrid[i][j].sick_days = 1;
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

    // Open the thread grid file in truncate mode initially to clear previous results
    std::ofstream thread_file("thread_grid.txt", std::ios::trunc);
    thread_file.close();

    double start_time = omp_get_wtime();

    /* Allocate grid and tracking arrays. Having grid and newGrid allows for:
      1. Consistent calculations by ensuring each day’s updates are based on a static snapshot of the previous day.
      2. Correct modeling of simultaneous infection / recovery without interference from updates that have not yet been fully applied.
    */
    Person** grid = new Person * [gridHeight];
    Person** newGrid = new Person * [gridHeight];
    int** threadGrid = new int* [gridHeight]; // Debug grid for tracking thread assignments
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = new Person[gridWidth];
        newGrid[i] = new Person[gridWidth];
        threadGrid[i] = new int[gridWidth];
    }

    initializeGrid(grid);
    printGridToFile(grid, 0);
    printThreadGridToFile(threadGrid, 0);

    // Simulation loop
    for (int day = 1; day <= numDays; ++day) {
        // Update the grid based on transmission/recovery rules
        updateGrid(grid, newGrid, threadGrid);

        // Print grid to file
        printGridToFile(newGrid, day);
        printThreadGridToFile(threadGrid, day);

        // Swap pointers instead of copying
        Person** temp = grid;
        grid = newGrid;
        newGrid = temp;
    }

    // Free dynamically allocated memory
    for (int i = 0; i < gridHeight; ++i) {
        delete[] grid[i];
        delete[] newGrid[i];
        delete[] threadGrid[i];
    }
    delete[] grid;
    delete[] newGrid;
    delete[] threadGrid;

    double end_time = omp_get_wtime();
    std::cout << "Simulation completed in: " << (end_time - start_time) << " seconds." << std::endl;

    return 0;
}