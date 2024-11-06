﻿#include <iostream>
#include <fstream>
#include <omp.h>

// Default values for simulation settings
int gridHeight = 5;
int gridWidth = 5;
double alpha = 0.1;
double beta = 0.3;
int omega = 2;
int numDays = 5;
int numThreads;

// Class to represent a person
class Person {
public:
    bool was_infected;
    int sick_days;

    Person() : was_infected(false), sick_days(0) {}
};

// Function to read settings from a configuration file
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

// Function to print the grid to a file
void printGridToFile(Person** grid, int day) {
    std::ofstream file("flu_simulation.txt", std::ios::app);
    file << "Day " << day << ":\n";
    file << "Grid (Infection Status):\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << (grid[i][j].sick_days > 0 ? 1 : 0) << " ";
        }
        file << "\n";
    }
    file << "\n";
    file.close();
}

// Update grid based on transmission and recovery rules
void updateGrid(Person** grid, Person** newGrid) {
#pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            newGrid[i][j] = grid[i][j]; // Copy current state

            if (grid[i][j].sick_days > 0) {  // If person is sick
                newGrid[i][j].sick_days++;  // Increment sick days
                if (newGrid[i][j].sick_days >= omega) {
                    newGrid[i][j].sick_days = 0;  // Person recovers
                }
            }
            else if (!grid[i][j].was_infected) {  // Only never-infected individuals
                int sickNeighbors = 0;
                if (i > 0 && grid[i - 1][j].sick_days > 0) sickNeighbors++;
                if (i < gridHeight - 1 && grid[i + 1][j].sick_days > 0) sickNeighbors++;
                if (j > 0 && grid[i][j - 1].sick_days > 0) sickNeighbors++;
                if (j < gridWidth - 1 && grid[i][j + 1].sick_days > 0) sickNeighbors++;

                unsigned int thread_seed = 123456789 + omp_get_thread_num(); // thread-specific seed
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

    double start_time = omp_get_wtime();
    // Allocate grid and tracking arrays
    Person** grid = new Person * [gridHeight];
    Person** newGrid = new Person * [gridHeight];
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = new Person[gridWidth];
        newGrid[i] = new Person[gridWidth];
    }

    initializeGrid(grid);
    printGridToFile(grid, 0);

    // Simulation loop
    for (int day = 1; day <= numDays; ++day) {
        // Update the grid based on transmission/recovery rules
        updateGrid(grid, newGrid);

        // Print grid to file
        printGridToFile(newGrid, day);

        // Swap pointers instead of copying
        Person** temp = grid;
        grid = newGrid;
        newGrid = temp;
    }

    // Free dynamically allocated memory
    for (int i = 0; i < gridHeight; ++i) {
        delete[] grid[i];
        delete[] newGrid[i];
    }
    delete[] grid;
    delete[] newGrid;

    double end_time = omp_get_wtime();
    std::cout << "Simulation completed in: " << (end_time - start_time) << " seconds." << std::endl;

    return 0;
}